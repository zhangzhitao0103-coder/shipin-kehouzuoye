"""逐句生成配音 → build/voice/Lxx.wav

优先使用真人录音：如果 voice_real/Lxx.wav（或 .mp3/.m4a）存在，直接用它；
否则用离线 TTS（sherpa-onnx + Kokoro 中文男声）生成占位配音。

用法: python3 tts.py --model-dir /path/to/kokoro-multi-lang-v1_0
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
SR = 48000


def trim(x, sr, pad=0.04):
    """去掉首尾静音，保留一点余量（阈值随音量自适应，兼容手机录音的底噪）"""
    thresh = max(0.005, 0.04 * np.abs(x).max())
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * sr))
    b = min(len(x), idx[-1] + int(pad * sr))
    return x[a:b]


def to_48k_mono(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", "1", "-ar", str(SR), str(dst)], check=True)


def load_tts(model_dir, threads):
    import sherpa_onnx
    m = Path(model_dir)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(m / "model.onnx"),
                voices=str(m / "voices.bin"),
                tokens=str(m / "tokens.txt"),
                data_dir=str(m / "espeak-ng-data"),
                dict_dir=str(m / "dict"),
                lexicon=f"{m / 'lexicon-us-en.txt'},{m / 'lexicon-zh.txt'}",
            ),
            num_threads=threads,
        ),
        rule_fsts=f"{m / 'date-zh.fst'},{m / 'phone-zh.fst'},{m / 'number-zh.fst'}",
        max_num_sentences=1,
    )
    return sherpa_onnx.OfflineTts(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=False)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    script = json.loads((HERE / "script.json").read_text())
    out = HERE / "build" / "voice"
    out.mkdir(parents=True, exist_ok=True)
    real = HERE / "voice_real"
    tts = None

    for line in script["lines"]:
        lid = line["id"]
        dst = out / f"{lid}.wav"
        rec = next((p for ext in ("wav", "mp3", "m4a") if (p := real / f"{lid}.{ext}").exists()), None)
        if rec:
            tmp = out / f"{lid}.tmp.wav"
            to_48k_mono(rec, tmp)
            x, sr = sf.read(tmp)
            tmp.unlink()
            src = "真人录音"
        else:
            if tts is None:
                if not args.model_dir:
                    raise SystemExit(f"{lid} 没有真人录音，需要 --model-dir 指向 Kokoro 模型来生成占位配音")
                tts = load_tts(args.model_dir, args.threads)
            v = script["voice"]
            a = tts.generate(line.get("say", line["sub"]), sid=v["speaker_id"],
                             speed=line.get("speed", v["speed"]))
            tmp = out / f"{lid}.tmp.wav"
            sf.write(tmp, np.array(a.samples, dtype=np.float32), a.sample_rate)
            to_48k_mono(tmp, dst)
            tmp.unlink()
            x, sr = sf.read(dst)
            src = "AI占位"
        x = trim(x, sr)
        sf.write(dst, x.astype(np.float32), sr)
        print(f"{lid} [{src}] {len(x) / sr:.2f}s  {line['sub']}")


if __name__ == "__main__":
    main()
