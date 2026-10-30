# 课后作业动画讲解视频 · 生成流水线

把「课后作业 + 解析 + 直播课」做成竖版（1080×1920）逐题动画讲解视频。

画面为机打排版（KaTeX 公式）逐步打出，配数轴动画、圈画标注、AI 配音和字幕；关键处插入直播课里老师的**课堂原声**，播放时右上角切换为老师的实时画面（人像抠图）。

## 流程

```
直播视频 ──► scripts/asr.py   语音转写，找老师讲方法的原话和时间点
          └► scripts/pip.py   截取片段，逐帧人像分割和头部跟踪，输出圆形画中画帧
脚本 specs/qN.json ──► make.py
      ├─ 逐句 TTS 配音（sherpa-onnx + Kokoro 中文音色），按句精确对齐字幕
      ├─ 计算时间轴：动作用「念到哪句话」做锚点，自动对齐配音
      ├─ render.js：Playwright 逐帧渲染 player.html → anim.mp4
      └─ ffmpeg：叠加老师画中画、混音、响度归一 → final.mp4
```

## 每道题的脚本（specs/qN.json）

- `problem`：题干（`$...$` 为公式，`\htmlClass{z1}{...}` 给局部打标记，方便圈画）
- `segments`：按顺序排列的片段
  - `tts`：`lines` 为 `[念的文字, 字幕显示]`，`actions` 为画面动作
  - `teacher`：`src` 为直播视频里的起止秒数，`lines` 为原声字幕，`pip` 为画中画帧目录
- 动作类型：`badge` 步骤标题，`line` 公式行（打字机效果），`html` 卡片，`nl` 数轴，`nlmark` 数轴标记（中间点或划掉），`nlmove` 动点和距离弧线，`hl` 圈出，`ul` 下划线，`tag` 上方标注，`st` 删除线
- `at` 和 `to` 填锚点文字，即动作在配音念到这几个字时开始或结束；也可以直接填片段内的秒数
- `tempo`：整体播放倍速（如 `1.5`）。配音和课堂原声做保调变速，动画按同样倍速逐帧重新采样，老师画中画同步加速，不会掉帧或错位

## 短视频模板：promo.py（如入营测答案速览）

- `specs/rucheng.json` → `promo.html`（1080×2400 安卓竖屏）：开场 → 逐题卡片（答案盖章、正确选项变绿、知识点逐个弹出、「体验课揭晓」引导条）→ 结尾引流页
- 场景字段：`intro`（`cards`）、`q`（`problem`/`options`/`answer`/`points`/`teaser`/`stars`/`qsize`）、`outro`（`headline`/`items`）；`actions` 用 `sel` + `fx`（`pop`/`up`/`slide`/`stamp`/`right`）+ `at` 锚点
- 背景音乐：`music.py` 用代码合成（120 BPM，C–G–Am–F），不涉及版权；老师说话时自动压低（sidechain），整体两遍线性响度归一到约 -15 LUFS，音乐不会在停顿处忽大忽小
- 老师头像：`scripts/cutout.py` 从直播笔记第 3 页官方照抠图，成品在 `assets/`（使用时复制到 `$WORK/promo/`）
- `voice.py`：两个模板共用的配音模块（Kokoro / 老师声音克隆 + 语音识别质检）

## 老师声音克隆配音（specs 中的 `clone` 字段）

- 模型：ZipVoice 零样本声音克隆（sherpa-onnx，`sherpa-onnx-zipvoice-zh-en-emilia`，full 模型，16 步）
- 参考音：直播课 33:45–33:53，老师讲解「x 一定在二和六的中间」的一段，配上它的准确文本（`clone.src` / `clone.text`）
- 注意：Aug-2025 的模型包里 `pinyin.raw` 与 sherpa-onnx 2.x 不兼容，汉字会被全部丢掉、合成出乱码。需要改用 Dec-2025 的 `sherpa-onnx-zipvoice-distill-int8-zh-en-emilia` 包里的 `lexicon.txt`/`tokens.txt`（权重仍可用 fp32 版）
- 字母处理：模型会丢掉中文句子里的拉丁字母，所以念的文字里自动替换为 x→艾克斯、a→诶、y→歪、D→滴
- 质检：每句合成后用语音识别回听，按字错率和「句尾是否完整」验收；不合格时放慢语速重试，结果写在 `out_*/clone_check.txt`
- 合规：按《人工智能生成合成内容标识办法》，模仿真人的合成语音须显著标识。画面左上方常驻「讲解配音为 AI 合成」提示，字幕标签为「AI合成配音」，与「课堂原声」区分。使用老师音色前须取得老师本人授权

## 运行环境

- Python：`sherpa-onnx soundfile numpy onnxruntime opencv-python-headless`
- Node：`npm i`（katex、@fontsource/noto-sans-sc），以及 playwright 和 chromium
- 模型从 GitHub Releases 下载到 `$WORK/models/`：
  - k2-fsa/sherpa-onnx：`sherpa-onnx-zipvoice-zh-en-emilia`、`sherpa-onnx-zipvoice-distill-int8-zh-en-emilia`（放在 `zv_int8/` 下）、`vocos_24khz.onnx`（声音克隆），`kokoro-multi-lang-v1_1`（备用 TTS）、`sherpa-onnx-paraformer-zh-2024-03-09`、`silero_vad.onnx`（ASR）
  - danielgatis/rembg：`u2net_human_seg.onnx`（人像分割）
- 直播视频放在 `$WORK/video/default.mp4`

```bash
export WORK=/path/to/work
python3 -I scripts/asr.py $WORK/models $WORK/video/audio16k.wav $WORK/transcript.json
python3 -I scripts/pip.py $WORK/models/u2net_human_seg.onnx $WORK/video/default.mp4 2085.8 2096.2 $WORK/pip_q1
python3 -I make.py specs/q1.json out_q1            # 加 --plan 只生成配音和时间轴
python3 -I promo.py specs/rucheng.json out_rc       # 入营测答案速览短视频
node snap.js out_q1/timeline.json out_q1/snap 12 30 60   # 抽帧预览版式
```

## 注意

- 不用克隆时（去掉 `clone` 字段）回退到 Kokoro 配音：`voice` 12 为女声，65 为男声；字母 a 会自动改写成「 A 」，避免被吞音。
- 课堂原声的字幕由语音识别生成，发布前请人工核对一遍。
