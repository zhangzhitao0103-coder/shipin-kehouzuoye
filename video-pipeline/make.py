# usage: WORK=/path/to/workdir python3 -I make.py spec.json outdir [--plan]
# WORK holds models/, video/default.mp4, pip_*/ (teacher frames from scripts/pip.py) and tts_cache/
import sys, os, json, subprocess, numpy as np, soundfile as sf

B = os.path.dirname(os.path.abspath(__file__))
S = os.path.abspath(os.environ.get('WORK') or os.path.dirname(B))
FPS, GAP = 25, 0.35

spec = json.load(open(sys.argv[1]))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
sys.path.insert(0, B)
import voice
voice.configure(S, spec, out)
tts, teacher_audio, rms_norm = voice.say, voice.teacher_audio, voice.rms_norm
SR = voice.SR

t = spec.get('lead', 0.8)
audio = [np.zeros(int(t * SR), np.float32)]
subs, actions, teacher = [], [], []
for si, seg in enumerate(spec['segments']):
    t0 = t; chunks = []
    if seg['type'] == 'tts':
        for spoken, shown in seg['lines']:
            x = rms_norm(tts(spoken)); d = len(x) / SR
            chunks.append((t, d, spoken)); subs.append({'t': t, 'e': t + d + 0.12, 'text': shown, 'who': seg.get('who', spec.get('who', '讲解')), 'ai': bool(voice.CLONE), 'k': len(subs)})
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
TEMPO = spec.get('tempo', 1.0)   # playback speed: >1 speeds narration, animation and teacher clips alike
audio.append(np.zeros(int(spec.get('tail', 1.2) * SR), np.float32))
wav = np.concatenate(audio); wav = np.clip(wav, -1, 1)
sf.write(f'{out}/audio.wav', wav, SR)
tl = {'fps': FPS, 'duration': dur, 'tempo': TEMPO, 'qno': spec['qno'], 'qsrc': spec.get('qsrc', ''), 'problem': spec['problem'],
      'pip': 'file://' + os.path.join(S, spec['pip_still']), 'note': spec.get('note', ''), 'actions': merged, 'subs': subs, 'teacher': teacher}
json.dump(tl, open(f'{out}/timeline.json', 'w'), ensure_ascii=False, indent=1)
print('duration %.1fs (%.1fs at %.2gx)' % (dur, dur / TEMPO, TEMPO))
if '--plan' in sys.argv: sys.exit()

# render animation frames
subprocess.run(['node', f'{B}/render.js', f'{out}/timeline.json', f'{out}/anim.mp4'], check=True, cwd=B)
# teacher picture-in-picture during original-audio clips
cmd = ['ffmpeg', '-v', 'error', '-y', '-i', f'{out}/anim.mp4', '-i', f'{out}/audio.wav']
fc, last = [], '0:v'
for i, tc in enumerate(teacher):
    t0, t1 = tc['t'] / TEMPO, tc['e'] / TEMPO
    cmd += ['-framerate', '%g' % (FPS * TEMPO), '-itsoffset', '%.3f' % t0, '-i', os.path.join(tc['pip'], 'p_%05d.png')]
    fc.append(f'[{i+2}:v]scale=228:228,format=rgba[p{i}]')
    fc.append(f"[{last}][p{i}]overlay=798:28:eof_action=pass:enable='between(t,{t0:.3f},{t1:.3f})'[v{i}]")
    last = f'v{i}'
if fc: cmd += ['-filter_complex', ';'.join(fc), '-map', f'[{last}]']
else: cmd += ['-map', '0:v']
cmd += ['-map', '1:a', '-c:v', 'libx264', '-crf', '19', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        '-af', ('atempo=%g,' % TEMPO if TEMPO != 1 else '') + 'loudnorm=I=-16:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-shortest', '-movflags', '+faststart', f'{out}/final.mp4']
subprocess.run(cmd, check=True)
print('done', f'{out}/final.mp4')
