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
