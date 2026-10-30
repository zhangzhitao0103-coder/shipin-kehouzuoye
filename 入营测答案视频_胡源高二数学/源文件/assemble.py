"""Assemble voice track, timeline.js, SFX + ducked BGM mix, and SRT."""
import json, os, sys
import numpy as np, soundfile as sf, librosa
from timeline import layout, write_js
import bgm as B

FPS = 30
SR = B.SR
here = os.path.dirname(os.path.abspath(__file__))
vdir = sys.argv[1] if len(sys.argv) > 1 else 'voice'
out = sys.argv[2] if len(sys.argv) > 2 else 'out'
os.makedirs(out, exist_ok=True)
sc = json.load(open(os.path.join(here, 'script.json')))

clips = []
for k in range(sum(len(s['lines']) for s in sc['scenes'])):
    x, sr = sf.read(f'{vdir}/l{k:02d}.wav', dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    clips.append(librosa.resample(x, orig_sr=sr, target_sr=SR, res_type='soxr_hq'))
durs = [len(c) / SR for c in clips]
scenes, total = layout(sc, durs)
n = int(total * SR) + SR
voice = np.zeros(n, dtype=np.float32)
k = 0
for s in scenes:
    for l in s['lines']:
        i = int(l['start'] * SR); voice[i:i + len(clips[k])] += clips[k]; k += 1
voice = voice[: int(total * SR)]
voice /= np.max(np.abs(voice)) + 1e-9
voice *= 0.89

# per-frame voice level for the avatar ring
hop = SR // FPS
nfr = int(np.ceil(total * FPS))
rms = np.array([np.sqrt(np.mean(voice[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(nfr)])
lv = np.clip((20 * np.log10(rms + 1e-9) + 42) / 26, 0, 1)
sm = np.zeros_like(lv); a = 0
for i, v in enumerate(lv):
    a = v if v > a else a * 0.7 + v * 0.3; sm[i] = a
write_js(os.path.join(here, 'timeline.js'), scenes, total, [round(float(v), 3) for v in sm], FPS)

# SFX cue list
sfx = np.zeros((len(voice), 2))
W, D, P, ST, RI = B.whoosh(), B.ding(), B.pop(), B.stamp(), B.rise(0.9)
cues = []
for i, s in enumerate(scenes):
    L = s['lines']
    if i: cues.append((W, s['vs'] - 0.12, 0.55))
    if s['id'].startswith('q'):
        tA = L[0]['start'] + (L[0]['end'] - L[0]['start']) * .45
        cues += [(D, tA, 0.42), (ST, tA, 0.45), (P, L[1]['start'], 0.5)]
    if s['id'] == 'intro':
        cues += [(ST, s['vs'] + .2, 0.6)] + [(P, L[1]['start'] + j * .16, 0.3) for j in range(4)] + [(P, L[2]['start'] + .1, 0.45)]
    if s['id'] == 'tease':
        cues += [(ST, s['vs'] + .05, 0.5), (RI, L[0]['start'] + (L[0]['end'] - L[0]['start']) * .55 - .55, 0.5), (D, L[0]['start'] + (L[0]['end'] - L[0]['start']) * .55 + .3, 0.35), (ST, L[1]['start'] + .7, 0.45)]
    if s['id'] == 'cta':
        cues += [(P, L[0]['start'] + (L[0]['end'] - L[0]['start']) * (.45 + j * .15), 0.4) for j in range(3)] + [(ST, L[2]['start'], 0.6), (D, L[2]['start'] + .05, 0.4)]
for x, t, g in cues:
    B.place(sfx, x, t, g)

music = B.make_bgm(total)[: len(voice)]
# ducking from voice envelope (attack 30ms / release 300ms)
env = np.abs(voice)
fr = SR // 100
e = np.array([env[i:i + fr].max() for i in range(0, len(env), fr)])
g = np.zeros_like(e); a = 0
for i, v in enumerate(e):
    a = a + (v - a) * (0.7 if v > a else 0.033); g[i] = a
g = np.clip(g / 0.3, 0, 1)
duck = np.repeat(1 - 0.6 * g, fr)[: len(voice)]
music *= (0.20 * duck)[:, None]

mix = music + sfx * 0.55 + voice[:, None]
peak = np.max(np.abs(mix)); mix /= max(peak, 1.0) / 0.97
sf.write(f'{out}/mix.wav', mix, SR)
sf.write(f'{out}/voice.wav', voice, SR)
inst = music + sfx * 0.55
sf.write(f'{out}/bgm_sfx.wav', inst / (np.max(np.abs(inst)) / 0.9), SR)

# SRT
def ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f'{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}'.replace(',1000', ',999')
with open(f'{out}/subtitles.srt', 'w') as f:
    allL = [l for s in scenes for l in s['lines']]
    for i, l in enumerate(allL):
        end = l['end'] + .2 if i + 1 == len(allL) else min(l['end'] + .2, allL[i + 1]['start'] - .02)
        f.write(f"{i + 1}\n{ts(l['start'])} --> {ts(end)}\n{l['cap'].replace('{', '').replace('}', '')}\n\n")
json.dump({'total': total, 'scenes': scenes}, open(f'{out}/timeline.json', 'w'), ensure_ascii=False, indent=1)
print('total', total)
for s in scenes: print(s['id'], s['vs'], s['lines'][0]['start'], s['lines'][-1]['end'])
