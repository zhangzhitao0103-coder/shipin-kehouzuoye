# Procedurally synthesised upbeat background loop (no third-party samples, so no licensing questions).
# usage: python3 -I music.py out.wav seconds [bpm]
import sys, numpy as np, soundfile as sf

SR = 44100
out, dur = sys.argv[1], float(sys.argv[2]); bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 120
beat = 60 / bpm; bar = 4 * beat
n = int((dur + 2) * SR); L = np.zeros(n); R = np.zeros(n)
rng = np.random.default_rng(7)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
# C - G - Am - F, voiced around middle C
CHORDS = [[60, 64, 67], [59, 62, 67], [57, 60, 64], [57, 60, 65]]
ROOTS = [36, 43, 45, 41]

def add(sig, t, pan=0.0, gain=1.0):
    i = int(t * SR); j = min(n, i + len(sig))
    if i >= n: return
    s = sig[:j - i] * gain
    L[i:j] += s * (1 - pan) ** 0.5; R[i:j] += s * (1 + pan) ** 0.5

def pluck(f, d=0.45):
    t = np.arange(int(d * SR)) / SR
    env = np.exp(-t * 9) * np.minimum(1, t * 400)
    return env * (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t))

def pad(fs, d):
    t = np.arange(int(d * SR)) / SR
    env = np.minimum(1, t / 0.35) * np.minimum(1, (d - t) / 0.4)
    s = sum(np.sin(2 * np.pi * f * k * t * (1 + 0.002 * k)) / k for f in fs for k in (1, 2, 3))
    return env * s / len(fs)

def bass(f, d):
    t = np.arange(int(d * SR)) / SR
    env = np.exp(-t * 6) * np.minimum(1, t * 300)
    return env * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t))

def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 50 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)

def hat():
    t = np.arange(int(0.06 * SR)) / SR
    x = rng.standard_normal(len(t)); x = np.diff(np.concatenate([[0], x]))   # crude high-pass
    return x * np.exp(-t * 70)

def clap():
    t = np.arange(int(0.18 * SR)) / SR
    x = rng.standard_normal(len(t)); x = x - np.convolve(x, np.ones(8) / 8, 'same')
    return x * np.exp(-t * 22)

nbars = int(dur / bar) + 2
for b in range(nbars):
    t0 = b * bar; c = CHORDS[b % 4]; full = b >= 2      # 2-bar soft intro, then the groove
    add(pad([hz(m) for m in c], bar), t0, gain=0.10)
    arp = [c[0], c[1], c[2], c[1] + 12, c[2], c[1], c[0] + 12, c[2]]
    for k, m in enumerate(arp):
        add(pluck(hz(m + 12)), t0 + k * beat / 2, pan=(-0.4 if k % 2 else 0.4), gain=0.22)
    if full:
        for k in range(8): add(bass(hz(ROOTS[b % 4] + (12 if k % 4 == 3 else 0)), beat / 2), t0 + k * beat / 2, gain=0.30)
        for k in range(4): add(kick(), t0 + k * beat, gain=0.55)
        for k in range(4): add(hat(), t0 + k * beat + beat / 2, pan=0.3, gain=0.045)
        for k in (1, 3): add(clap(), t0 + k * beat, gain=0.06)
# simple stereo echo for space
d = int(beat * 0.75 * SR)
L[d:] += 0.25 * R[:-d]; R[d:] += 0.25 * L[:-d]
x = np.stack([L, R], 1)[:int(dur * SR)]
F = np.fft.rfftfreq(len(x), 1 / SR)   # gentle low-pass above ~7 kHz keeps the top end out of the voice's way
x = np.fft.irfft(np.fft.rfft(x, axis=0) * (1 / np.sqrt(1 + (F / 7000) ** 4))[:, None], n=len(x), axis=0)

fo = int(1.5 * SR); x[-fo:] *= np.linspace(1, 0, fo)[:, None]
x /= np.abs(x).max() / 0.8
sf.write(out, x.astype(np.float32), SR)
print('music', out, '%.1fs' % dur)
