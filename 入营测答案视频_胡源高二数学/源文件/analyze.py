import sys, glob, os, numpy as np, soundfile as sf, librosa, sherpa_onnx
P = os.environ.get('PARAFORMER_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sherpa-onnx-paraformer-zh-small-2024-03-09'))
rec = sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=f'{P}/model.int8.onnx', tokens=f'{P}/tokens.txt', num_threads=4)
def asr(path):
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000); sr = 16000
    s = rec.create_stream(); s.accept_waveform(sr, x); rec.decode_stream(s); return s.result.text
def f0(path):
    x, sr = librosa.load(path, sr=16000)
    f, v, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr)
    f = f[~np.isnan(f)]
    return np.median(f), np.std(np.log2(f)) * 12
if __name__ == '__main__':
    for p in sys.argv[1:]:
        m, sd = f0(p) if '--nof0' not in sys.argv else (0, 0)
        print(os.path.basename(p), f'F0med={m:.0f}Hz spread={sd:.1f}st', asr(p))
