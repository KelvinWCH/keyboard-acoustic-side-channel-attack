import argparse
import sounddevice as sd
import numpy as np
import os
import wave
import time
import sys


SAMPLE_RATE = 44100
KEYS = list("abcdefghijklmnopqrstuvwxyz0123456789")
PRESSES_PER_KEY = 25
RECORD_SECONDS = 30
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def save_wav(filepath, audio, sample_rate):
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def list_audio_devices():
    print("\nAvailable input devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']} (inputs: {d['max_input_channels']})")
    print()


def record_key(key, device=None):
    print(f"\n{'='*50}")
    print(f"Key: '{key}'")
    print(f"Press it {PRESSES_PER_KEY} times, about 1 sec apart")
    print(f"You have {RECORD_SECONDS} seconds")
    print(f"{'='*50}")

    input("Press ENTER when ready to start recording...")

    print("RECORDING -- start pressing!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )

    for remaining in range(RECORD_SECONDS, 0, -1):
        sys.stdout.write(f"\r  Time remaining: {remaining:2d}s ")
        sys.stdout.flush()
        time.sleep(1)

    sd.wait()
    print("\nDone!")

    return audio.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== ASCA Keystroke Recorder ===")
    print(f"36 keys x 25 presses each")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")

    list_audio_devices()

    device_input = input("Enter device index (or ENTER for default): ").strip()
    device = int(device_input) if device_input else None

    # resume if we've already recorded some keys
    already_done = set()
    for f in os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []:
        if f.endswith(".wav"):
            already_done.add(f.replace(".wav", ""))

    remaining_keys = [k for k in KEYS if k not in already_done]

    if already_done:
        print(f"\nFound {len(already_done)} keys already recorded, "
              f"resuming with {len(remaining_keys)} remaining.")

    for i, key in enumerate(remaining_keys):
        print(f"\nProgress: {len(already_done) + i + 1}/{len(KEYS)} keys")

        audio = record_key(key, device=device)

        filepath = os.path.join(args.output_dir, f"{key}.wav")
        save_wav(filepath, audio, SAMPLE_RATE)
        print(f"Saved -> {filepath}")

        redo = input("Redo this key? (y/N): ").strip().lower()
        if redo == "y":
            audio = record_key(key, device=device)
            save_wav(filepath, audio, SAMPLE_RATE)
            print(f"Re-saved -> {filepath}")

    print(f"\nAll {len(KEYS)} keys recorded!")
    print(f"Files in: {os.path.abspath(args.output_dir)}")
    print("Next step: python ../2_preprocess/isolate.py")


if __name__ == "__main__":
    main()
