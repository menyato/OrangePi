#!/usr/bin/env bash
# probe_devices.sh — run on OrangePi to list every audio and video device
# Usage:  bash probe_devices.sh 2>&1 | tee probe_out.txt
#         Then paste probe_out.txt here so we can read the right ports.

SEP="════════════════════════════════════════════════════════════"

echo "$SEP"
echo "  ALSA CARDS  (cat /proc/asound/cards)"
echo "$SEP"
cat /proc/asound/cards 2>/dev/null || echo "  (not available)"

echo ""
echo "$SEP"
echo "  ALSA PLAYBACK DEVICES  (aplay -l)"
echo "$SEP"
aplay -l 2>/dev/null || echo "  (aplay not found)"

echo ""
echo "$SEP"
echo "  ALSA CAPTURE DEVICES  (arecord -l)"
echo "$SEP"
arecord -l 2>/dev/null || echo "  (arecord not found)"

echo ""
echo "$SEP"
echo "  ALSA PLAYBACK DEVICES VERBOSE  (aplay -L | grep -A1 'plughw\|hw:')"
echo "$SEP"
aplay -L 2>/dev/null | grep -E "^(plughw|hw:|default|sysdefault)" || echo "  (none)"

echo ""
echo "$SEP"
echo "  SOUNDDEVICE PYTHON LIST  (python3)"
echo "$SEP"
python3 - << 'PYEOF'
import sounddevice as sd
devs = sd.query_devices()
for i, d in enumerate(devs):
    tag = []
    if d["max_input_channels"]  > 0: tag.append(f"IN ch={d['max_input_channels']}")
    if d["max_output_channels"] > 0: tag.append(f"OUT ch={d['max_output_channels']}")
    print(f"  [{i:2d}] {d['name']:<45} {' | '.join(tag)}")
print(f"\n  Default input  → {sd.query_devices(kind='input')['name']}")
print(f"  Default output → {sd.query_devices(kind='output')['name']}")
PYEOF

echo ""
echo "$SEP"
echo "  V4L2 VIDEO NODES  (v4l2-ctl --list-devices)"
echo "$SEP"
v4l2-ctl --list-devices 2>/dev/null || echo "  (v4l2-ctl not installed — run: sudo apt install v4l-utils)"

echo ""
echo "$SEP"
echo "  V4L2 CARD NAMES PER NODE  (/dev/video*)"
echo "$SEP"
for node in $(ls /dev/video* 2>/dev/null | sort -V); do
    info=$(v4l2-ctl --device "$node" --info 2>/dev/null \
           | grep -E "Card type|Driver name|Bus info" \
           | sed 's/^\s*//')
    if [ -n "$info" ]; then
        echo "  $node:"
        echo "$info" | sed 's/^/      /'
    else
        echo "  $node:  (no v4l2 info)"
    fi
done

echo ""
echo "$SEP"
echo "  QUICK ALSA SPEAKER TEST — plughw:3,0 and plughw:4,0"
echo "  (each plays a 440 Hz tone for 1 second)"
echo "$SEP"
for dev in plughw:3,0 plughw:4,0; do
    echo -n "  Testing $dev ... "
    # Generate a raw 440 Hz sine wave (16-bit, mono, 44100 Hz, 1 sec) and pipe to aplay
    python3 -c "
import struct, math
rate=44100; freq=440; dur=1
samples=[int(32000*math.sin(2*math.pi*freq*t/rate)) for t in range(rate*dur)]
import sys; sys.stdout.buffer.write(struct.pack('<'+'h'*len(samples),*samples))
" 2>/dev/null | aplay -D "$dev" -f S16_LE -r 44100 -c 1 -q 2>&1 \
        && echo "OK" || echo "FAILED"
done

echo ""
echo "$SEP"
echo "  QUICK MIC TEST — record 3 s from device index 2, print RMS"
echo "$SEP"
python3 - << 'PYEOF'
import sounddevice as sd, numpy as np
MIC_INDEX = 2
RATE      = 16000
SECS      = 3
try:
    print(f"  Recording {SECS}s from device [{MIC_INDEX}]...")
    audio = sd.rec(int(SECS * RATE), samplerate=RATE,
                   channels=1, dtype="float32", device=MIC_INDEX)
    sd.wait()
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"  RMS={rms:.5f}  peak={peak:.5f}")
    if rms < 0.0001:
        print("  WARNING: signal very low — mic may be muted or wrong device")
    else:
        print("  Mic looks alive.")
except Exception as e:
    print(f"  ERROR: {e}")
PYEOF

echo ""
echo "$SEP"
echo "  DONE — paste the output above back into the chat"
echo "$SEP"
