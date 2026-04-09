import argparse
import numpy as np
import os
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 44100
DEFAULT_ISOLATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "isolated")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "spectrograms")

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = 224


def wav_to_melspec(filepath):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def save_spectrogram_image(mel_spec_db, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(2.24, 2.24), dpi=100)
    ax.imshow(mel_spec_db, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_ISOLATED_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Spectrogram Generator ===")
    print(f"Input:  {os.path.abspath(args.input_dir)}")
    print(f"Output: {os.path.abspath(args.output_dir)}\n")

    if not os.path.exists(args.input_dir):
        print(f"Error: {args.input_dir} not found. Run isolate.py first.")
        return

    key_dirs = sorted(os.listdir(args.input_dir))
    total = 0

    for key in key_dirs:
        key_input = os.path.join(args.input_dir, key)
        key_output = os.path.join(args.output_dir, key)

        if not os.path.isdir(key_input):
            continue

        os.makedirs(key_output, exist_ok=True)

        # clear old PNGs from previous runs
        for f in os.listdir(key_output):
            if f.endswith(".png"):
                os.remove(os.path.join(key_output, f))

        wav_files = sorted([f for f in os.listdir(key_input) if f.endswith(".wav")])
        count = 0

        for wav_file in wav_files:
            wav_path = os.path.join(key_input, wav_file)
            png_name = wav_file.replace(".wav", ".png")
            png_path = os.path.join(key_output, png_name)

            mel_spec_db = wav_to_melspec(wav_path)
            save_spectrogram_image(mel_spec_db, png_path)
            count += 1

        print(f"  Key '{key}': {count} spectrograms")
        total += count

    print(f"\nGenerated {total} spectrogram images")
    print(f"Saved to: {os.path.abspath(args.output_dir)}")
    print("Next step: python ../3_train/train.py")


if __name__ == "__main__":
    main()
