import argparse
import numpy as np
import os
import wave
import librosa


SAMPLE_RATE = 44100
DEFAULT_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "isolated")

WINDOW_MS = 30
HOP_MS = 5
MIN_GAP_MS = 200
CLIP_DURATION_MS = 100
TARGET_PRESSES = 25


def load_wav(filepath):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    return audio


def save_wav(filepath, audio):
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def compute_energy(audio, window_size, hop_size):
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        frame = audio[i : i + window_size]
        energy.append(np.sum(frame ** 2))
    return np.array(energy)


def find_keystroke_peaks(energy, threshold, min_gap_frames):
    peaks = []
    i = 0
    while i < len(energy):
        if energy[i] > threshold:
            burst_start = i
            while i < len(energy) and energy[i] > threshold * 0.5:
                i += 1
            burst_end = i
            peak_idx = burst_start + int(np.argmax(energy[burst_start:burst_end]))
            peaks.append(peak_idx)
            i = burst_end + min_gap_frames
        else:
            i += 1
    return peaks


def find_adaptive_peaks(energy, target, min_gap_frames, iterations=30):
    # binary search the threshold until we get close to `target` detections
    lo = max(np.percentile(energy, 50), 1e-12)
    hi = max(np.max(energy), lo * 2)
    best = find_keystroke_peaks(energy, lo, min_gap_frames)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        peaks = find_keystroke_peaks(energy, mid, min_gap_frames)
        if abs(len(peaks) - target) < abs(len(best) - target):
            best = peaks
        if len(peaks) == target:
            return peaks
        if len(peaks) > target:
            lo = mid
        else:
            hi = mid
    return best


def extract_clips(audio, peaks, hop_size, clip_samples):
    clips = []
    for peak in peaks:
        sample_idx = peak * hop_size
        start = max(0, sample_idx - clip_samples // 4)
        end = start + clip_samples
        if end <= len(audio):
            clips.append(audio[start:end])
    return clips


def isolate_keystrokes(audio, target=TARGET_PRESSES):
    window_size = int(SAMPLE_RATE * WINDOW_MS / 1000)
    hop_size = int(SAMPLE_RATE * HOP_MS / 1000)
    min_gap_frames = int(MIN_GAP_MS / HOP_MS)
    clip_samples = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)

    energy = compute_energy(audio, window_size, hop_size)
    peaks = find_adaptive_peaks(energy, target, min_gap_frames)
    return extract_clips(audio, peaks, hop_size, clip_samples)


def process_key(key, raw_path, output_dir, target):
    key_dir = os.path.join(output_dir, key)
    os.makedirs(key_dir, exist_ok=True)

    # wipe old clips so we don't mix them with the new ones
    for f in os.listdir(key_dir):
        if f.endswith(".wav"):
            os.remove(os.path.join(key_dir, f))

    audio = load_wav(raw_path)
    clips = isolate_keystrokes(audio, target=target)

    print(f"  Key '{key}': detected {len(clips)} keystrokes")

    for i, clip in enumerate(clips[:target]):
        filepath = os.path.join(key_dir, f"{key}_{i+1:02d}.wav")
        save_wav(filepath, clip)

    return len(clips[:target])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", type=int, default=TARGET_PRESSES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Keystroke Isolation ===")
    print(f"Input:  {os.path.abspath(args.raw_dir)}")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print(f"Target: {args.target} presses per key\n")

    if not os.path.exists(args.raw_dir):
        print(f"Error: {args.raw_dir} not found. Run record.py first.")
        return

    wav_files = sorted([f for f in os.listdir(args.raw_dir) if f.endswith(".wav")])
    print(f"Found {len(wav_files)} key recordings\n")

    total_clips = 0
    for f in wav_files:
        key = f.replace(".wav", "")
        raw_path = os.path.join(args.raw_dir, f)
        count = process_key(key, raw_path, args.output_dir, args.target)
        total_clips += count

    print(f"\nIsolated {total_clips} keystroke clips")
    print(f"Saved to: {os.path.abspath(args.output_dir)}")
    print("Next step: python spectrogram.py")


if __name__ == "__main__":
    main()
