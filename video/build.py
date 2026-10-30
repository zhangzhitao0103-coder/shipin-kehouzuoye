"""根据逐句配音的时长排时间轴，并合成整条音轨（配音 + 背景音乐 + 音效）

输入: script.json, build/voice/Lxx.wav（先跑 tts.py）
输出: build/timeline.json  画面动画用的时间轴
      build/mix.wav        最终音轨
      ../output/字幕.srt
背景音乐和音效全部由代码合成，无版权问题。
"""
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
BUILD = HERE / "build"
SR = 48000

LEAD = 0.45        # 片头留白
LINE_GAP = 0.12    # 同一场景内句间停顿
SCENE_GAP = 0.5    # 换场景停顿（转场动画）
TAIL = 2.2         # 片尾定格


# ---------------------------------------------------------------- timeline

def split_caption(text, max_len=15):
    """按标点切成短字幕，去掉句中标点（保留问号/感叹号）"""
    parts = [p for p in re.split(r"(?<=[，。；：、！？,])", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > max_len:
            chunks.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    return [re.sub(r"[，。；：、,]", " ", c).strip() for c in chunks]


def build_timeline(script):
    lines, scenes, cues, caps = {}, {}, {}, []
    t, prev_scene = LEAD, None
    for line in script["lines"]:
        x, _ = sf.read(BUILD / "voice" / f"{line['id']}.wav")
        dur = len(x) / SR
        if prev_scene and line["scene"] != prev_scene:
            t += SCENE_GAP - LINE_GAP
        start, end = t, t + dur
        lines[line["id"]] = {"start": start, "end": end, "scene": line["scene"], "sub": line["sub"]}
        sc = scenes.setdefault(line["scene"], {"start": start, "end": end})
        sc["end"] = end
        for c in line.get("cues", []):
            cues[c["name"]] = {"t": start + c["at"] * dur, "sfx": c.get("sfx")}
        chunks = split_caption(line["sub"])
        total = sum(len(c) for c in chunks)
        ct = start
        for c in chunks:
            d = dur * len(c) / total
            caps.append({"start": ct, "end": ct + d, "text": c})
            ct += d
        prev_scene = line["scene"]
        t = end + LINE_GAP
    duration = t - LINE_GAP + TAIL
    names = list(scenes)
    for a, b in zip(names, names[1:]):
        scenes[a]["out"] = scenes[b]["start"] - 0.25
    scenes[names[-1]]["out"] = duration + 1
    return {"duration": duration, "fps": script["fps"], "size": script["size"],
            "lines": lines, "scenes": scenes, "cues": cues, "captions": caps}


def write_srt(caps, path):
    def ts(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    path.write_text("\n".join(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n"
                              for i, c in enumerate(caps, 1)), encoding="utf-8")


# ---------------------------------------------------------------- synthesis

def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def tone(freq, dur, harm=(1.0,), tau=None, attack=0.005):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harm))
    e = np.minimum(1, t / attack) if attack else np.ones(n)
    if tau:
        e = e * env_exp(n, tau)
    return y * e


def add(buf, sig, t):
    i = int(t * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += sig


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def make_bgm(duration, drums_from):
    """轻快的 I–V–vi–IV 进行，116 BPM：pad + 拨弦琶音 + 贝斯 + 轻鼓"""
    rng = np.random.default_rng(7)
    bpm = 116
    beat = 60 / bpm
    bar = 4 * beat
    n = int((duration + 1) * SR)
    pad, arp, bass, drum = (np.zeros(n) for _ in range(4))
    chords = [(60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)]  # C G Am F
    nbars = int(duration / bar) + 1
    kick = tone(1, 0.3)
    tk = np.arange(len(kick)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 75 * np.exp(-tk / 0.04)) / SR) * env_exp(len(kick), 0.11)
    snare = lowpass(rng.standard_normal(int(0.2 * SR)), 3500) * env_exp(int(0.2 * SR), 0.05)
    hn = rng.standard_normal(int(0.06 * SR))
    hat = (hn - lowpass(hn, 6000)) * env_exp(len(hn), 0.012)
    for b in range(nbars):
        t0 = b * bar
        ch = chords[b % 4]
        for m in ch:  # pad：慢起慢收
            for det in (0.998, 1.002):
                s = tone(midi(m) * det, bar + 0.3, harm=(1, 0.25, 0.08), attack=0.25)
                s *= np.minimum(1, (len(s) - np.arange(len(s))) / (0.35 * SR))
                add(pad, s, t0)
        pattern = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[0] + 24, ch[2] + 12, ch[1] + 12, ch[0] + 24, ch[2] + 12]
        for k, m in enumerate(pattern):
            add(arp, tone(midi(m), 0.45, harm=(1, 0.3, 0.1), tau=0.16), t0 + k * beat / 2)
        if t0 >= drums_from - 0.01:
            for k in range(4):
                add(bass, tone(midi(ch[0] - 24), beat * 0.95, harm=(1, 0.35), tau=0.3, attack=0.01), t0 + k * beat)
                if k in (0, 2):
                    add(drum, kick, t0 + k * beat)
                else:
                    add(drum, snare * 0.35, t0 + k * beat)
                add(drum, hat * 0.25, t0 + k * beat + beat / 2)
    mix = 0.10 * pad + 0.16 * arp + 0.30 * bass + 0.55 * drum
    return mix[: int(duration * SR)] / (np.abs(mix).max() + 1e-9)


def make_sfx(kind):
    rng = np.random.default_rng(3)
    if kind == "ding":       # 答对：清脆双音
        out = np.zeros(int(0.9 * SR))
        add(out, 0.6 * tone(midi(88), 0.9, harm=(1, 0.2), tau=0.25), 0)
        add(out, 0.5 * tone(midi(95), 0.8, harm=(1, 0.2), tau=0.25), 0.07)
        return out
    if kind == "pop":        # 标签弹出
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        return 0.5 * np.sin(2 * np.pi * np.cumsum(500 + 700 * np.exp(-t / 0.02)) / SR) * env_exp(n, 0.03)
    if kind == "whoosh":     # 转场
        n = int(0.45 * SR)
        e = np.sin(np.pi * np.arange(n) / n) ** 2
        return 0.35 * lowpass(rng.standard_normal(n), 1800) * e
    if kind == "strike":     # 划掉
        n = int(0.25 * SR)
        return 0.4 * lowpass(rng.standard_normal(n), 2500) * env_exp(n, 0.06)
    if kind == "rise":       # 大招登场：上行琶音
        out = np.zeros(int(0.9 * SR))
        for k, m in enumerate((72, 76, 79, 84)):
            add(out, 0.45 * tone(midi(m), 0.5, harm=(1, 0.3), tau=0.18), k * 0.08)
        return out
    if kind == "chime":      # 片尾：和弦铃声
        out = np.zeros(int(1.6 * SR))
        for k, m in enumerate((79, 84, 88, 91)):
            add(out, 0.35 * tone(midi(m), 1.4, harm=(1, 0.25), tau=0.45), k * 0.05)
        return out
    raise ValueError(kind)


def mix_audio(tl):
    dur = tl["duration"]
    n = int(dur * SR)
    voice = np.zeros(n)
    active = np.zeros(n)
    for lid, l in tl["lines"].items():
        x, _ = sf.read(BUILD / "voice" / f"{lid}.wav")
        rms = np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
        x = x * (10 ** (-17 / 20) / rms)         # 每句响度对齐到约 -17 dBFS
        add(voice, x, l["start"])
        add(active, np.ones(len(x)), l["start"])
    # 人声出现时把背景音乐压低（平滑 ducking）
    k = int(0.15 * SR)
    duck = np.convolve(np.minimum(active, 1), np.ones(k) / k, mode="same")
    bgm_gain = 10 ** (-24 / 20) * (1 - duck) + 10 ** (-31 / 20) * duck
    bgm = make_bgm(dur, drums_from=tl["scenes"]["q1"]["start"] - 0.3)
    fade = np.minimum(1, (n - np.arange(n)) / (1.8 * SR))      # 结尾淡出
    fade *= np.minimum(1, np.arange(n) / (0.3 * SR))             # 开头淡入
    bgm = bgm * bgm_gain * fade

    sfx = np.zeros(n)
    for name, sc in tl["scenes"].items():
        if name != "intro":
            add(sfx, make_sfx("whoosh"), sc["start"] - 0.45)
    for c in tl["cues"].values():
        if c["sfx"]:
            add(sfx, make_sfx(c["sfx"]), c["t"])
    sfx *= 10 ** (-15 / 20)

    out = voice + bgm + sfx
    peak = np.abs(out).max()
    if peak > 0.97:
        out *= 0.97 / peak
    return out.astype(np.float32)


def main():
    script = json.loads((HERE / "script.json").read_text())
    tl = build_timeline(script)
    (BUILD / "timeline.json").write_text(json.dumps(tl, ensure_ascii=False, indent=1))
    out_dir = HERE.parent / "output"
    out_dir.mkdir(exist_ok=True)
    write_srt(tl["captions"], out_dir / "字幕.srt")
    sf.write(BUILD / "mix.wav", mix_audio(tl), SR)
    print(f"总时长 {tl['duration']:.1f}s，{len(tl['captions'])} 条字幕")
    for name, sc in tl["scenes"].items():
        print(f"  {name:6s} {sc['start']:5.2f} → {sc['end']:5.2f}")


if __name__ == "__main__":
    main()
