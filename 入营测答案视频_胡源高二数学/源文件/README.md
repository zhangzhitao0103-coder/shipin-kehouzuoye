# 源文件：入营测答案揭晓视频

画面是一个 1080×1920 的 HTML 页面（`page.html`），用 Playwright 按 30fps 逐帧截图；
配音、背景音乐、音效在 Python 里合成和混音；最后用 ffmpeg 合成 MP4。

## 文件

| 文件 | 作用 |
|---|---|
| `script.json` | 文案：每个场景的每一句（`cap` 字幕，`{}` 内为橙色高亮；`tts` 朗读文本） |
| `tts_gen.py` / `tts_lib.py` | 离线中文 TTS（sherpa-onnx + Kokoro v1.1-zh），每句生成 `voice/lNN.wav`，并用 ASR 回听校验 |
| `analyze.py` | ASR（Paraformer）回听 + 音高分析，用于校验发音 |
| `timeline.py` | 根据每句时长排时间轴（句间 0.08s，场景间 0.40s） |
| `assemble.py` | 拼接人声、生成 `timeline.js`、合成音效与 BGM（人声自动压低 BGM）、输出 `out/mix.wav` 与 `out/subtitles.srt` |
| `bgm.py` | 纯代码合成的原创背景音乐和音效（无版权素材） |
| `page.html` | 画面与动画（KaTeX 渲染公式，SVG 示意图） |
| `render.js` | Playwright 逐帧截图 |
| `assets/teacher.png` | 胡源老师照片（取自直播笔记 PDF 第 3 页） |

## 重新生成

```bash
pip install sherpa-onnx soundfile librosa numpy
npm install
```

模型放到 `models/`（或用环境变量 `KOKORO_DIR` / `PARAFORMER_DIR` 指定）：

```bash
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2 && tar xjf kokoro-int8-multi-lang-v1_1.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2 && tar xjf sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2
cd ..
```

1. 配音（成片使用音色 90、语速 1.28；第 3、14 句（`l02`、`l13`）用的是语速 1.2 的版本，发音更清楚）：

```bash
python3 tts_gen.py 90 1.28 voice
```

   **换成老师真声**：把录音按句放成 `voice/l00.wav` … `voice/l22.wav`，跳过这一步即可。

2. 时间轴 + 混音 + 字幕：

```bash
python3 assemble.py voice out
```

3. 逐帧渲染（可开多个进程分段渲染，总帧数 = 时长 × 30）：

```bash
mkdir -p frames && node render.js frames frames 30 0 1815
```

4. 响度标准化（-14 LUFS）并合成：

```bash
ffmpeg -y -i out/mix.wav -af "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000" out/mix_norm.wav
```

```bash
ffmpeg -y -framerate 30 -i frames/f%05d.jpg -i out/mix_norm.wav -map 0:v -map 1:a -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -profile:v high -level 4.1 -r 30 -g 60 -c:a aac -b:a 192k -ar 48000 -movflags +faststart -shortest out/final.mp4
```

预览单帧：`node render.js preview prev 12.5 30`（输出 `prev/t12.50.png`）。
