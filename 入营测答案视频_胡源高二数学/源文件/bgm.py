"""Procedural, copyright-free upbeat BGM + SFX (numpy only)."""
import numpy as np, soundfile as sf, sys

SR = 44100
BPM = 124
BEAT = 60 / BPM
rng = np.random.default_rng(7)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def env_ad(n, a, d):
    t = np.arange(n) / SR
    e = np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-np.maximum(t - a, 0) / d)
    return e


def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    s = 0.0
    for i in range(len(x)):  # small buffers only
        s = (1 - a) * x[i] + a * s
        y[i] = s
    return y


def lp_fft(x, fc, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / fc) ** (2 * order))
    return np.fft.irfft(X, len(x))


def hp_fft(x, fc, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X, len(x))


def bp_fft(x, lo, hi):
    return hp_fft(lp_fft(x, hi), lo)


def saw(freq, n, detune=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in (-detune, 0, detune) if detune else (0,):
        f = freq * 2 ** (d / 1200)
        ph = rng.random()
        out += 2 * ((t * f + ph) % 1.0) - 1
    return out / (3 if detune else 1)


def kick():
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = 45 + 95 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / 0.13) * 0.95


def clap():
    n = int(0.22 * SR)
    x = bp_fft(rng.standard_normal(n), 900, 4000)
    e = np.zeros(n)
    for k, o in enumerate((0, 0.011, 0.022)):
        i = int(o * SR)
        e[i:] += env_ad(n - i, 0.001, 0.012 if k < 2 else 0.07)
    return x * e * 0.35


def hat(open_=False):
    n = int((0.16 if open_ else 0.05) * SR)
    x = hp_fft(rng.standard_normal(n), 7000)
    return x * env_ad(n, 0.0008, 0.06 if open_ else 0.014) * 0.22


def pluck(freq, dur=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = (saw(freq, n) * 0.5 + np.sin(2 * np.pi * freq * t) * 0.5)
    x = lp_fft(x, 3200)
    return x * env_ad(n, 0.002, 0.09)


def bell(freq, dur=1.0, amp=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * freq * t) + 0.45 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t / 0.08)
         + 0.25 * np.sin(2 * np.pi * freq * 5.4 * t) * np.exp(-t / 0.04))
    return x * env_ad(n, 0.002, 0.35) * amp


def place(buf, x, t0, gain=1.0, pan=0.0):
    i = int(round(t0 * SR))
    if i >= buf.shape[0] or i + len(x) <= 0:
        return
    j = min(buf.shape[0], i + len(x))
    seg = x[: j - i] * gain
    buf[i:j, 0] += seg * np.sqrt(0.5 * (1 - pan))
    buf[i:j, 1] += seg * np.sqrt(0.5 * (1 + pan))


# F  G  Em  Am  (IV V iii vi in C) -- chord tones as MIDI
CHORDS = [
    [53, 57, 60, 65],  # F
    [55, 59, 62, 67],  # G
    [52, 55, 59, 64],  # Em
    [57, 60, 64, 69],  # Am
]
ENDING = [48, 52, 55, 60]  # C


def make_bgm(total):
    bar = 4 * BEAT
    nbars = int(np.ceil(total / bar)) + 1
    n = int((nbars * bar + 2) * SR)
    drums = np.zeros((n, 2)); bass = np.zeros((n, 2)); pad = np.zeros((n, 2)); arp = np.zeros((n, 2))
    K, C, H, HO = kick(), clap(), hat(), hat(True)
    last_bar = int(total // bar)
    for b in range(nbars):
        t0 = b * bar
        ch = ENDING if b >= last_bar else CHORDS[b % 4]
        intro = b < 1
        # drums
        for beat in range(4):
            tb = t0 + beat * BEAT
            if not intro:
                place(drums, K, tb, 0.8)
            if beat in (1, 3) and not intro:
                place(drums, C, tb, 0.9, 0.05)
            place(drums, H if beat % 2 == 0 else HO, tb + BEAT / 2, 0.8 if not intro else 0.5, 0.3)
            place(drums, H, tb + BEAT * 0.25, 0.35, -0.3)
            place(drums, H, tb + BEAT * 0.75, 0.35, -0.3)
        # bass: offbeat-pumping 8ths, root
        root = midi(ch[0] - 12)
        for k in range(8):
            if intro and k < 4:
                continue
            nn = int(BEAT / 2 * SR * 0.9)
            tt = np.arange(nn) / SR
            f = root * (2 if k in (3, 7) else 1)
            x = np.sin(2 * np.pi * f * tt) * 0.8 + 0.25 * lp_fft(saw(f, nn), 600)
            x *= env_ad(nn, 0.005, 0.12)
            place(bass, x, t0 + k * BEAT / 2, 0.5)
        # pad
        nn = int(bar * SR)
        x = sum(saw(midi(m), nn, detune=12) for m in ch) / len(ch)
        x = lp_fft(x, 1400)
        tt = np.arange(nn) / SR
        x *= np.minimum(tt / 0.25, 1) * np.minimum((bar - tt) / 0.2, 1)
        place(pad, x, t0, 0.28, -0.2)
        place(pad, np.roll(x, 300), t0, 0.28, 0.2)
        # arp 16ths
        seq = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[3] + 12, ch[2] + 12, ch[1] + 12, ch[3] + 12, ch[2] + 24]
        for k in range(16):
            if intro and k % 2:
                continue
            place(arp, pluck(midi(seq[k % 8])), t0 + k * BEAT / 4, 0.18, 0.35 if k % 2 else -0.35)
    # sidechain pump on bass/pad from kick grid
    tgrid = np.arange(n) / SR
    phase = (tgrid % BEAT) / BEAT
    pump = 1 - 0.55 * np.exp(-phase / 0.18)
    pump[: int(bar * SR)] = 1.0
    bass *= pump[:, None]; pad *= pump[:, None]
    # simple stereo delay on arp
    d = int(BEAT * 0.75 * SR)
    arp_d = np.zeros_like(arp); arp_d[d:, 0] = arp[:-d, 1] * 0.35; arp_d[d:, 1] = arp[:-d, 0] * 0.35
    mix = drums * 0.9 + bass + pad + arp + arp_d
    mix = mix[: int(total * SR)]
    # fade in/out
    fi = int(0.4 * SR); fo = int(2.5 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
    mix /= np.max(np.abs(mix)) + 1e-9
    return mix * 0.9


def whoosh(dur=0.45):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    t = np.arange(n) / SR
    out = np.zeros(n)
    # sweep band by chunks
    chunks = 24
    for c in range(chunks):
        a, b = c * n // chunks, (c + 1) * n // chunks
        fc = 400 + 5000 * np.sin(np.pi * c / chunks) ** 2
        seg = lp_fft(x[a:b], fc)
        out[a:b] = seg
    e = np.sin(np.pi * t / dur) ** 2
    return out * e * 0.5


def ding():
    x = bell(midi(84), 1.2, 0.5)
    y = np.zeros(len(x) + int(0.09 * SR))
    y[: len(x)] += x
    y[int(0.09 * SR):] += bell(midi(91), 1.2, 0.45)
    return y


def pop():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f = 600 + 900 * np.exp(-t / 0.02)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(n, 0.001, 0.03) * 0.5


def stamp():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * np.cumsum(90 + 120 * np.exp(-t / 0.02)) / SR) * env_ad(n, 0.001, 0.06)
    click = hp_fft(rng.standard_normal(n), 2500) * env_ad(n, 0.0005, 0.008)
    return body * 0.7 + click * 0.4


def rise(dur=1.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 200 * 2 ** (3 * t / dur)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.15 + hp_fft(rng.standard_normal(n), 3000) * 0.1 * (t / dur)
    return x * (t / dur) ** 2


if __name__ == '__main__':
    total = float(sys.argv[1]); out = sys.argv[2]
    sf.write(out, make_bgm(total), SR)
