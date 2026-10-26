# usage: WORK=/path/to/workdir python3 -I make.py spec.json outdir [--plan]
# WORK holds models/, video/default.mp4, pip_*/ (teacher frames from scripts/pip.py) and tts_cache/
import sys, os, json, hashlib, subprocess, numpy as np, soundfile as sf

B = os.path.dirname(os.path.abspath(__file__))
S = os.path.abspath(os.environ.get('WORK') or os.path.dirname(B))
M = os.path.join(S, 'models')
VIDEO = os.path.join(S, 'video', 'default.mp4')
SR, FPS, GAP = 24000, 25, 0.35

spec = json.load(open(sys.argv[1]))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
cache = os.path.join(S, 'tts_cache'); os.makedirs(cache, exist_ok=True)
VOICE, SPEED = spec.get('voice', 12), spec.get('speed', 1.05)

_tts = None
import re
def tts(text):
    global _tts
    text = re.sub(r'(?<![A-Za-z])a(?![A-Za-z])', ' A ', text)   # lone "a" is swallowed by the voice model
    h = hashlib.md5(f'{VOICE}|{SPEED}|{text}'.encode()).hexdigest()[:16]
    f = os.path.join(cache, h + '.wav')
    if not os.path.exists(f):
        if _tts is None:
            import sherpa_onnx
            K = f'{M}/kokoro-multi-lang-v1_1'
            cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f'{K}/model.onnx', voices=f'{K}/voices.bin', tokens=f'{K}/tokens.txt', data_dir=f'{K}/espeak-ng-data',
                dict_dir=f'{K}/dict', lexicon=f'{K}/lexicon-us-en.txt,{K}/lexicon-zh.txt'), num_threads=4),
                rule_fsts=f'{K}/date-zh.fst,{K}/phone-zh.fst,{K}/number-zh.fst')
            _tts = sherpa_onnx.OfflineTts(cfg)
        a = _tts.generate(text, sid=VOICE, speed=SPEED)
        assert a.sample_rate == SR
        x = np.array(a.samples, np.float32)
        nz = np.where(np.abs(x) > 0.01)[0]          # trim leading/trailing silence
        if len(nz): x = x[max(0, nz[0] - 600): nz[-1] + 1200]
        sf.write(f, x, SR)
    x, _ = sf.read(f, dtype='float32'); return x

def teacher_audio(s, e):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(s), '-t', str(e - s), '-i', VIDEO, '-ac', '1', '-ar', str(SR),
                          '-af', 'highpass=f=80,afade=t=in:d=0.08,afade=t=out:st=%f:d=0.15' % (e - s - 0.15), '-f', 'f32le', '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def rms_norm(x, target=0.08):
    r = np.sqrt(np.mean(x[np.abs(x) > 0.005] ** 2)) if np.any(np.abs(x) > 0.005) else 1
    return x * (target / max(r, 1e-4))

t = spec.get('lead', 0.8)
audio = [np.zeros(int(t * SR), np.float32)]
subs, actions, teacher = [], [], []
for si, seg in enumerate(spec['segments']):
    t0 = t; chunks = []
    if seg['type'] == 'tts':
        for spoken, shown in seg['lines']:
            x = rms_norm(tts(spoken)); d = len(x) / SR
            chunks.append((t, d, spoken)); subs.append({'t': t, 'e': t + d + 0.12, 'text': shown, 'who': seg.get('who', '讲解'), 'k': len(subs)})
            audio += [x, np.zeros(int(0.18 * SR), np.float32)]; t += d + 0.18
    else:
        s, e = seg['src']; x = rms_norm(teacher_audio(s, e), 0.085); d = len(x) / SR
        teacher.append({'t': t, 'e': t + d, 'src': [s, e], 'pip': os.path.join(S, seg['pip'])})
        for a, b, shown in seg['lines']:
            subs.append({'t': t + a, 'e': t + b, 'text': shown, 'who': '胡源老师 · 课堂原声', 'teacher': True, 'k': len(subs)})
        audio.append(x); t += d
    def when(at):
        if isinstance(at, (int, float)): return t0 + at
        for (c0, d, sp) in chunks:
            i = sp.find(at)
            if i >= 0: return c0 + d * i / max(1, len(sp))
        raise SystemExit(f'anchor not found in segment {si}: {at}')
    for a in seg.get('actions', []):
        a = dict(a); a['t'] = when(a.get('at', 0))
        if 'until' in a and a['until'] is not None: a['until'] = when(a['until'])
        if 'to' in a: a['d'] = max(0.3, when(a['to']) - a['t'])
        if a['do'] == 'nl':
            for p in a.get('pts', []): p['t'] = when(p.get('at', a.get('at', 0)))
            for m in a.get('marks', []): m['t'] = when(m['at'])
            if 'mover' in a: a['mover']['t'] = when(a['mover']['at'])
        actions.append(a)
    pad = seg.get('after', GAP); audio.append(np.zeros(int(pad * SR), np.float32)); t += pad

# actions that modify an existing number line (marks/mover added later) -> merge into the nl definition
merged = []
for a in actions:
    if a['do'] in ('nlmark', 'nlmove'):
        base = next(x for x in merged if x['do'] == 'nl' and x['id'] == a['nl'])
        if a['do'] == 'nlmark': base.setdefault('marks', []).append({**a['mark'], 't': a['t']})
        else: base['mover'] = {**a['mover'], 't': a['t']}
    else: merged.append(a)

dur = t + spec.get('tail', 1.2)
audio.append(np.zeros(int(spec.get('tail', 1.2) * SR), np.float32))
wav = np.concatenate(audio); wav = np.clip(wav, -1, 1)
sf.write(f'{out}/audio.wav', wav, SR)
tl = {'fps': FPS, 'duration': dur, 'qno': spec['qno'], 'qsrc': spec.get('qsrc', ''), 'problem': spec['problem'],
      'pip': 'file://' + os.path.join(S, spec['pip_still']), 'actions': merged, 'subs': subs, 'teacher': teacher}
json.dump(tl, open(f'{out}/timeline.json', 'w'), ensure_ascii=False, indent=1)
print('duration %.1fs' % dur)
if '--plan' in sys.argv: sys.exit()

# render animation frames
subprocess.run(['node', f'{B}/render.js', f'{out}/timeline.json', f'{out}/anim.mp4'], check=True, cwd=B)
# teacher picture-in-picture during original-audio clips
cmd = ['ffmpeg', '-v', 'error', '-y', '-i', f'{out}/anim.mp4', '-i', f'{out}/audio.wav']
fc, last = [], '0:v'
for i, tc in enumerate(teacher):
    cmd += ['-framerate', str(FPS), '-itsoffset', '%.3f' % tc['t'], '-i', os.path.join(tc['pip'], 'p_%05d.png')]
    fc.append(f'[{i+2}:v]scale=228:228,format=rgba[p{i}]')
    fc.append(f"[{last}][p{i}]overlay=798:28:eof_action=pass:enable='between(t,{tc['t']:.3f},{tc['e']:.3f})'[v{i}]")
    last = f'v{i}'
if fc: cmd += ['-filter_complex', ';'.join(fc), '-map', f'[{last}]']
else: cmd += ['-map', '0:v']
cmd += ['-map', '1:a', '-c:v', 'libx264', '-crf', '19', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-shortest', '-movflags', '+faststart', f'{out}/final.mp4']
subprocess.run(cmd, check=True)
print('done', f'{out}/final.mp4')
