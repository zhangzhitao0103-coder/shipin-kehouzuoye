# Short promo video (e.g. 入营测 answer reveal): scenes of cards + teacher-voice narration + background music.
# usage: WORK=/path/to/workdir python3 -I promo.py spec.json outdir [--plan]
import sys, os, json, subprocess, numpy as np, soundfile as sf

B = os.path.dirname(os.path.abspath(__file__))
S = os.path.abspath(os.environ.get('WORK') or os.path.dirname(B))
spec = json.load(open(sys.argv[1]))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
sys.path.insert(0, B)
import voice
voice.configure(S, spec, out)
SR, FPS = voice.SR, 25
TEMPO = spec.get('tempo', 1.0)
LINE_GAP, SCENE_GAP = spec.get('line_gap', 0.12), spec.get('scene_gap', 0.35)

t = spec.get('lead', 0.5)
audio = [np.zeros(int(t * SR), np.float32)]
scenes, subs, actions = [], [], []
for si, sc in enumerate(spec['scenes']):
    t0 = t; chunks = []
    for spoken, shown in sc['lines']:
        x = voice.rms_norm(voice.say(spoken)); d = len(x) / SR
        chunks.append((t, d, spoken))
        subs.append({'t': t, 'e': t + d + 0.1, 'text': shown, 'who': spec.get('who', ''), 'k': len(subs)})
        audio += [x, np.zeros(int(LINE_GAP * SR), np.float32)]; t += d + LINE_GAP
    def when(at):
        if isinstance(at, (int, float)): return t0 + at
        for c0, d, sp in chunks:
            i = sp.find(at)
            if i >= 0: return c0 + d * i / max(1, len(sp))
        raise SystemExit(f'anchor not found in scene {si}: {at}')
    for a in sc.get('actions', []):
        actions.append({**a, 'scene': si, 't': when(a.get('at', 0)) + a.get('delay', 0)})
    hold = sc.get('hold', 0)   # extra time on screen after the last line (e.g. to let the outro breathe)
    audio.append(np.zeros(int((SCENE_GAP + hold) * SR), np.float32)); t += SCENE_GAP + hold
    scenes.append({k: v for k, v in sc.items() if k not in ('lines', 'actions')} | {'t': max(0, t0 - 0.3)})
for a, b in zip(scenes, scenes[1:] + [None]): a['e'] = b['t'] if b else t + 10
dur = t + spec.get('tail', 1.0)
audio.append(np.zeros(int(spec.get('tail', 1.0) * SR), np.float32))
sf.write(f'{out}/voice.wav', np.clip(np.concatenate(audio), -1, 1), SR)
tl = {'fps': FPS, 'duration': dur, 'tempo': TEMPO, 'size': spec.get('size', [1080, 2400]), 'page': 'promo.html',
      'avatar': 'file://' + os.path.join(S, spec['avatar']), 'portrait': 'file://' + os.path.join(S, spec['portrait']),
      'note': spec.get('note', ''), 'note_from': dur - spec['note_last'] * TEMPO if spec.get('note_last') else 0, 'scenes': scenes, 'actions': actions, 'subs': subs}
json.dump(tl, open(f'{out}/timeline.json', 'w'), ensure_ascii=False, indent=1)
print('duration %.1fs (%.1fs at %.2gx)' % (dur, dur / TEMPO, TEMPO))
if '--plan' in sys.argv: sys.exit()

final_len = dur / TEMPO
subprocess.run([sys.executable, '-I', f'{B}/music.py', f'{out}/music.wav', '%.2f' % (final_len + 0.5), str(spec.get('bpm', 120))], check=True)
subprocess.run(['node', f'{B}/render.js', f'{out}/timeline.json', f'{out}/anim.mp4'], check=True, cwd=B)
# voice sped up (pitch kept); music ducks under the voice via sidechain compression
fc = (f"[0:a]atempo={TEMPO},aresample=48000,asplit=2[v1][v2];"
      f"[1:a]aresample=48000,volume={spec.get('music_gain', 0.4)}[m];"
      "[m][v1]sidechaincompress=threshold=0.04:ratio=2.5:attack=20:release=600[md];"
      "[v2][md]amix=inputs=2:duration=first:normalize=0[a]")
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{out}/voice.wav', '-i', f'{out}/music.wav', '-filter_complex', fc,
                '-map', '[a]', '-c:a', 'pcm_f32le', f'{out}/mix.wav'], check=True)
# two-pass linear loudness normalisation: one fixed gain, so the music does not pump up in the pauses
LN = 'loudnorm=I=-14:TP=-1.5:LRA=11'
r = subprocess.run(['ffmpeg', '-hide_banner', '-i', f'{out}/mix.wav', '-af', LN + ':print_format=json', '-f', 'null', '-'],
                   capture_output=True, text=True).stderr
m = json.loads(r[r.rindex('{'):r.rindex('}') + 1])
ln2 = (LN + f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
       f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{out}/anim.mp4', '-i', f'{out}/mix.wav', '-map', '0:v', '-map', '1:a',
                '-af', ln2 + ',aresample=48000', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                '-shortest', '-movflags', '+faststart', f'{out}/final.mp4'], check=True)
print('done', f'{out}/final.mp4')
