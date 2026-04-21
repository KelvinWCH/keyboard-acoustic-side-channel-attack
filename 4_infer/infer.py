import os
import sys
import json
import time
import argparse
import queue
import numpy as np
import sounddevice as sd
import librosa
import torch
import timm
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from io import BytesIO
from PIL import Image


SAMPLE_RATE = 44100
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_SIZE = 224

THRESHOLD = 0.008
MIN_GAP_SEC = 0.15
CLIP_DURATION_SEC = 0.10

CORRECTION_BUFFER_SIZE = 1000
CORRECTION_PAUSE_SEC = 10.0


def load_model(device):
    model_path = os.path.join(MODEL_DIR, "coatnet_keystrokes.pth")
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    with open(mapping_path) as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    model = timm.create_model("coatnet_0_rw_224", pretrained=False, num_classes=36)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (val acc: {checkpoint['val_acc']:.2f}%)")
    return model, idx_to_class


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def clip_to_spectrogram(clip):
    mel_spec = librosa.feature.melspectrogram(
        y=clip, sr=SAMPLE_RATE,
        n_mels=128, n_fft=2048, hop_length=512,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(2.24, 2.24), dpi=100)
    ax.imshow(mel_spec_db, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert("RGB")


def correct_with_llm(raw_chars, ollama_host):
    prompt = (
        "The following characters were predicted from keyboard typing sounds "
        "using an acoustic side-channel attack. They may contain errors. "
        "Correct them into the most likely English text. "
        "Output ONLY the corrected text, nothing else.\n\n"
        f"Input: \"{raw_chars}\""
    )
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "gemma4:e2b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=60,
        )
        result = response.json()
        return result.get("response", raw_chars).strip()
    except Exception as e:
        return f"[LLM error: {e}] {raw_chars}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--device-index", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== ASCA Live Inference ===\n")

    print("Audio devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            dflt = " <--" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{dflt}")
    print()

    model, idx_to_class = load_model(device)
    transform = get_inference_transform()

    ollama_ok = False
    try:
        r = requests.get(f"{args.ollama_host}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any("gemma4" in m for m in models):
            print("Ollama: connected")
            ollama_ok = True
        else:
            print(f"Ollama: gemma4 not found. Available: {models}")
    except Exception:
        print(f"Ollama: not reachable at {args.ollama_host}")

    print(f"\nThreshold: {THRESHOLD}")
    print(f"Clip duration: {CLIP_DURATION_SEC}s")
    print(f"Min gap: {MIN_GAP_SEC}s")
    print("\nListening, type some keys\n")

    raw_predictions = []
    last_keystroke_time = time.time()
    corrected_output = []
    clip_queue = queue.Queue()

    clip_samples = int(SAMPLE_RATE * CLIP_DURATION_SEC)
    min_gap_samples = int(SAMPLE_RATE * MIN_GAP_SEC)
    # put the keystroke at 25% of the clip so it matches training
    pre_peak_samples = clip_samples // 4
    post_peak_samples = clip_samples - pre_peak_samples
    ring_buffer = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    cb_state = {"ring_pos": 0, "total": 0, "last_det": 0}
    pending_peaks = []

    def audio_callback(indata, frames, time_info, status):
        chunk = indata.flatten()
        n = len(chunk)
        pos = cb_state["ring_pos"]

        end = pos + n
        if end <= len(ring_buffer):
            ring_buffer[pos:end] = chunk
        else:
            first = len(ring_buffer) - pos
            ring_buffer[pos:] = chunk[:first]
            ring_buffer[:n - first] = chunk[first:]
        new_pos = end % len(ring_buffer)

        chunk_start_total = cb_state["total"]
        cb_state["total"] += n

        rms = np.sqrt(np.mean(chunk ** 2))

        bar_len = int(min(rms / (THRESHOLD * 3), 1.0) * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        hit = " <<< KEYSTROKE" if rms > THRESHOLD else ""
        sys.stdout.write(f"\r  RMS: {rms:.6f} |{bar}| thr: {THRESHOLD}{hit}    ")
        sys.stdout.flush()

        if rms > THRESHOLD:
            gap = cb_state["total"] - cb_state["last_det"]
            if gap > min_gap_samples:
                peak_local = int(np.argmax(np.abs(chunk)))
                peak_global = chunk_start_total + peak_local
                pending_peaks.append(peak_global)
                cb_state["last_det"] = cb_state["total"]

        # wait until we have enough audio after the peak, then grab the clip
        while pending_peaks and cb_state["total"] >= pending_peaks[0] + post_peak_samples:
            peak_global = pending_peaks.pop(0)
            start = (peak_global - pre_peak_samples) % len(ring_buffer)
            end_idx = start + clip_samples
            if end_idx <= len(ring_buffer):
                clip = ring_buffer[start:end_idx].copy()
            else:
                first = len(ring_buffer) - start
                clip = np.concatenate([
                    ring_buffer[start:],
                    ring_buffer[:clip_samples - first]
                ])
            clip_queue.put(clip)

        cb_state["ring_pos"] = new_pos

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.05),
        callback=audio_callback,
        device=args.device_index,
    )

    try:
        with stream:
            while True:
                while not clip_queue.empty():
                    clip = clip_queue.get()

                    img = clip_to_spectrogram(clip)
                    tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(tensor)
                        probs = torch.softmax(output, dim=1)
                        confidence, predicted = probs.max(1)
                        key = idx_to_class[predicted.item()]

                    raw_predictions.append(key)
                    last_keystroke_time = time.time()

                    print(f"\n  key: '{key}' (conf: {confidence.item():.1%}) "
                          f"| so far: {''.join(raw_predictions)}")

                idle_time = time.time() - last_keystroke_time
                should_correct = (
                    len(raw_predictions) >= CORRECTION_BUFFER_SIZE
                    or (len(raw_predictions) > 0
                        and idle_time > CORRECTION_PAUSE_SEC)
                )

                if should_correct and raw_predictions:
                    raw_text = "".join(raw_predictions)
                    print(f"\n  sending to Gemma: \"{raw_text}\"")

                    if ollama_ok:
                        corrected = correct_with_llm(raw_text, args.ollama_host)
                    else:
                        corrected = raw_text + " (no LLM)"

                    corrected_output.append(corrected)
                    print(f"  corrected: \"{corrected}\"\n")
                    raw_predictions.clear()

                time.sleep(0.01)

    except KeyboardInterrupt:
        if raw_predictions:
            raw_text = "".join(raw_predictions)
            if ollama_ok:
                corrected = correct_with_llm(raw_text, args.ollama_host)
            else:
                corrected = raw_text
            corrected_output.append(corrected)
            print(f"\n  {raw_text} -> {corrected}")

        print("\n\nSession ended.")
        if corrected_output:
            print(f"Full text: {' '.join(corrected_output)}")
        else:
            print("No keystrokes detected.")
            print("If the RMS bar never spiked, your mic isn't picking up typing.")
            print("If it spiked but no detections, lower THRESHOLD in the script.")


if __name__ == "__main__":
    main()
