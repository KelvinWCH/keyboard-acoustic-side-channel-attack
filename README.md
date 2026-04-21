# Keyboard Acoustic Side-Channel Attack

A small end-to-end project that trains a CoAtNet image classifier to recognize
individual keys from the sound they make when pressed, then runs live inference
on microphone input.

The pipeline follows the recipe from Harrison et al. (2023) — record keystrokes,
turn each press into a mel-spectrogram image, fine-tune a pretrained CoAtNet-0,
and use the model as a real-time acoustic keylogger.

## Pipeline

```
1_collect/    record.py         raw audio, 25 presses per key
2_preprocess/ isolate.py        split each recording into per-keystroke clips
              spectrogram.py    convert clips to 224x224 mel-spectrogram PNGs
3_train/      train.py          fine-tune CoAtNet-0 on the spectrograms
4_infer/      infer.py          live mic inference
```

## Setup

```
pip install -r requirements.txt
```

Tested on Python 3.11 with CUDA 12 (RTX 4070). CPU also works but training is
much slower.

## Usage

Record 25 presses for each of the 36 keys (a-z, 0-9):

```
python 1_collect/record.py
```

Split into per-keystroke clips and generate spectrograms:

```
python 2_preprocess/isolate.py
python 2_preprocess/spectrogram.py
```

Train:

```
python 3_train/train.py
```

Run live inference:

```
python 4_infer/infer.py
```

### Held-out validation session

Random-split validation on a single recording session heavily over-estimates
accuracy because the train/val clips come from the same microphone position,
typing cadence, and ambient noise. To get a realistic number, record a second
session and use it for validation:

```
python 1_collect/record.py     --output-dir data/raw_holdout
python 2_preprocess/isolate.py  --raw-dir data/raw_holdout --output-dir data/isolated_holdout
python 2_preprocess/spectrogram.py --input-dir data/isolated_holdout --output-dir data/spectrograms_holdout
python 3_train/train.py         --val-dir data/spectrograms_holdout
```

## Method

- **Audio**: 44.1 kHz mono, 100 ms clips centered so the keystroke sits at
  ~25% of the window.
- **Features**: 128-band mel-spectrogram (log power, n_fft=2048, hop=512),
  rendered as a 224x224 PNG to match the CoAtNet input size.
- **Model**: CoAtNet-0 (`coatnet_0_rw_224` from `timm`), pretrained on
  ImageNet, fine-tuned in two phases — 10 epochs with the backbone frozen,
  then 54 epochs end-to-end at 0.1x LR.
- **Augmentation**: small horizontal translation + two RandomErasing passes
  (one blocky, one thin vertical — approximates SpecAugment on a rendered
  spectrogram).
- **Loss**: cross-entropy with label smoothing (0.1), AdamW, cosine LR schedule.
- **Isolation**: the isolator binary-searches an energy threshold per
  recording so it finds exactly ~25 presses regardless of volume variation
  between keys.
- **Live alignment**: the inference callback defers clip extraction until
  ~75 ms of post-peak audio has accumulated, so the keystroke lands at 25%
  of the window — matching the training alignment.

## Results

With a single recording session and a random 80/20 split, validation
accuracy tops out at ~81%. This number is optimistic: clips from the same
session are highly correlated, so the split measures memorization more than
generalization. Accuracy on a separately recorded session is substantially
lower — that's the number that actually matters for a real attack, and is
why the `--val-dir` flag exists.
