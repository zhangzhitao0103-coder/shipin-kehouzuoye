"""Generate per-line voice clips for script.json, trim silences, verify with ASR."""
import sys, os, json, re, difflib
import numpy as np, soundfile as sf
from tts_lib import make_tts
from analyze import asr

sid = int(sys.argv[1]); speed = float(sys.argv[2]); out = sys.argv[3]
os.makedirs(out, exist_ok=True)
sc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'script.json')))
tts = make_tts()
norm = lambda s: re.sub(r'[^\w]', '', s)
res = []
k = 0
for s in sc['scenes']:
    for l in s['lines']:
        a = tts.generate(l['tts'], sid=sid, speed=speed)
        x = np.array(a.samples, dtype=np.float32); sr = a.sample_rate
        # trim silence (keep 40 ms pad)
        env = np.convolve(np.abs(x), np.ones(240) / 240, mode='same')
        idx = np.where(env > 0.01 * np.max(env))[0]
        pad = int(0.04 * sr)
        x = x[max(0, idx[0] - pad): min(len(x), idx[-1] + pad)]
        p = f'{out}/l{k:02d}.wav'; sf.write(p, x, sr)
        hyp = asr(p)
        m = difflib.SequenceMatcher(None, norm(l['tts']), norm(hyp)).ratio()
        res.append({'k': k, 'dur': round(len(x) / sr, 3), 'match': round(m, 3), 'hyp': hyp})
        print(k, f'{len(x)/sr:.2f}s', f'{m:.2f}', l['tts'], '=>', hyp, flush=True)
        k += 1
json.dump(res, open(f'{out}/lines.json', 'w'), ensure_ascii=False, indent=1)
print('TOTAL speech', round(sum(r['dur'] for r in res), 2), 'mean match', round(np.mean([r['match'] for r in res]), 3))
