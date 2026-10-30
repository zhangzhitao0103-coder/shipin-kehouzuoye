import sherpa_onnx, soundfile as sf, numpy as np, os
M = os.environ.get('KOKORO_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'kokoro-int8-multi-lang-v1_1'))
def make_tts():
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f'{M}/model.int8.onnx', voices=f'{M}/voices.bin', tokens=f'{M}/tokens.txt',
                data_dir=f'{M}/espeak-ng-data', dict_dir=f'{M}/dict',
                lexicon=f'{M}/lexicon-us-en.txt,{M}/lexicon-zh.txt'),
            num_threads=4),
        rule_fsts=f'{M}/date-zh.fst,{M}/phone-zh.fst,{M}/number-zh.fst', max_num_sentences=1)
    return sherpa_onnx.OfflineTts(cfg)
def say(tts, text, sid, speed, path):
    a = tts.generate(text, sid=sid, speed=speed)
    x = np.array(a.samples, dtype=np.float32)
    sf.write(path, x, a.sample_rate)
    return len(x) / a.sample_rate
