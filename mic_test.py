# save as mic_test.py and run it
import sounddevice as sd
import numpy as np
import sys

print("Devices:")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        dflt = " <<<" if i == sd.default.device[0] else ""
        print(f"  [{i}] {d['name']}{dflt}")

print("\nListening for 10 sec - type some keys...\n")

def cb(indata, frames, t, status):
    rms = np.sqrt(np.mean(indata.flatten()**2))
    bars = int(min(rms * 500, 40))
    sys.stdout.write(f"\r  RMS: {rms:.6f} |{'#' * bars}{'.' * (40-bars)}|")
    sys.stdout.flush()

with sd.InputStream(samplerate=44100, channels=1, dtype="float32",
                     blocksize=2205, callback=cb):
    sd.sleep(10000)