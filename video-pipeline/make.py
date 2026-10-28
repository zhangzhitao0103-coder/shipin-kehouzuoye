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

CLONE = spec.get('clone')   # {"src": [s, e], "text": prompt transcript, "model": "full"|"distill", "steps": 16}
LETTERS = {'x': '艾克斯', 'a': '诶', 'y': '歪', 'b': '必', 'A': '诶', 'B': '必', 'C': '西', 'D': '滴'}
def zh_letters(t):  # the clone model drops Latin letters inside Chinese text; spell them the way teachers read them
    return re.sub(r'(?<![A-Za-z])([A-Za-z])(?![A-Za-z])', lambda m: LETTERS.get(m.group(1), m.group(1)), t)

_tts = _clone = _asr = None
import re
def tts(text):
    if CLONE: return clone(text)
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

def norm_cmp(t):  # text form used to compare TTS input with what speech recognition hears back
    for k, v in [('艾克斯', 'x'), ('诶', 'a'), ('歪', 'y'), ('滴', 'd'), ('儿', '')]: t = t.replace(k, v)
    return re.sub(r'[^\w]|_', '', t.lower())

def cer(ref, hyp):
    r, h = norm_cmp(ref), norm_cmp(hyp); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

def recognize(x):
    global _asr
    import sherpa_onnx
    if _asr is None:
        P = f'{M}/sherpa-onnx-paraformer-zh-2024-03-09'
        _asr = sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=f'{P}/model.int8.onnx', tokens=f'{P}/tokens.txt', num_threads=4)
    y = np.interp(np.arange(0, len(x), SR / 16000), np.arange(len(x)), x).astype(np.float32)
    s = _asr.create_stream(); s.accept_waveform(16000, y); _asr.decode_stream(s); return s.result.text

def clone(text):
    """Zero-shot clone of the teacher's voice (ZipVoice), with an ASR check and retries on mispronunciation."""
    global _clone
    text = zh_letters(text)
    model, steps = CLONE.get('model', 'full'), CLONE.get('steps', 16)
    h = hashlib.md5(f'clone|{CLONE["src"]}|{model}|{steps}|{text}'.encode()).hexdigest()[:16]
    f = os.path.join(cache, h + '.wav')
    if not os.path.exists(f):
        import sherpa_onnx
        if _clone is None:
            Z = f'{M}/sherpa-onnx-zipvoice-{"distill-" if model == "distill" else ""}zh-en-emilia'
            L = f'{M}/zv_int8/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia'   # Dec-2025 package: lexicon/tokens matching sherpa-onnx 2.x
            cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=f'{L}/tokens.txt', encoder=f'{Z}/text_encoder.onnx', decoder=f'{Z}/fm_decoder.onnx',
                vocoder=f'{M}/vocos_24khz.onnx', data_dir=f'{L}/espeak-ng-data', lexicon=f'{L}/lexicon.txt'), num_threads=4))
            s, e = CLONE['src']
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(s), '-t', str(e - s), '-i', VIDEO, '-ac', '1', '-ar', str(SR),
                                  '-af', 'highpass=f=70', '-f', 'f32le', '-'], capture_output=True).stdout
            _clone = (sherpa_onnx.OfflineTts(cfg), np.frombuffer(raw, np.float32).tolist(), zh_letters(CLONE['text']))
        engine, prompt, ptext = _clone
        best = None
        ref = norm_cmp(text)
        for speed in (1.0, 0.92, 0.85, 1.06):    # a slower speed gives the model more frames when it drops words
            a = engine.generate(text, ptext, prompt, SR, speed, steps)
            x = np.array(a.samples, np.float32)
            nz = np.where(np.abs(x) > 0.01)[0]
            if len(nz): x = x[max(0, nz[0] - 600): nz[-1] + 2400]
            heard = recognize(x); err = cer(text, heard)
            tail = len(ref) < 3 or ref[-2:] in norm_cmp(heard)[-6:]   # sentence endings must not be swallowed
            key = (not tail, err)
            if best is None or key < best[0]: best = (key, x, heard, speed)
            if tail and err <= 0.08: break
        (miss, err), x, heard, speed = best
        with open(os.path.join(out, 'clone_check.txt'), 'a') as log:
            log.write(f'CER={err:.2f}{" TAIL-MISSING" if miss else ""}\tspeed={speed}\t{text}\t=> {heard}\n')
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
            chunks.append((t, d, spoken)); subs.append({'t': t, 'e': t + d + 0.12, 'text': shown, 'who': seg.get('who', spec.get('who', '讲解')), 'ai': bool(CLONE), 'k': len(subs)})
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
