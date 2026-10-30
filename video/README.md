# 胡源 · 课前测答案速览视频

高二数学入营测（3 道函数最值题）的答案速览短视频：报答案 + 点考点 + 留"大招"钩子，引导进胡源老师体验课。

- 成片：`../output/胡源_课前测答案速览_1080x1920.mp4`（竖屏 9:16，30fps，H.264 + AAC，约 51 秒）
- 封面：`../output/封面.jpg`
- 字幕：`../output/字幕.srt`（画面里已经烧录了字幕，srt 供剪映等二次编辑用）
- 口播稿：`../output/口播稿.md`

## 结构

| 段落 | 时长 | 画面 |
|---|---|---|
| 开场 | ~5s | 老师头像 + "课前测试 答案揭晓" + 三道题编号 |
| 第 1 题 | ~9s | 题目 → 选项 A 亮绿 + 盖章 → 数轴标出 [1,2] → 考点卡片 + 思路提示 |
| 第 2 题 | ~5s | 题目 → "答案 x = 2" → 数轴 → 考点卡片 |
| 第 3 题 | ~9s | 题目 → 选项 B → 阿氏圆动画（A 沿圆移动到最高点）→ 考点卡片 |
| 大招悬念 | ~11s | 常规解法（草稿 + 秒表飞到 4:59）被划掉 → "胡源大招 10 秒出答案" → 两张上锁卡片"体验课揭晓" |
| 引流收尾 | ~7s | 老师照片 + 履历 + "高二数学体验课 · 我们直播间见" |

背景音乐和音效全部由 `build.py` 用代码合成（无版权风险），人声出现时自动压低音乐。

## 换成胡源老师真人配音

1. 让老师按 `../output/口播稿.md` 一句一个文件录音，命名 `L01.m4a` … `L16.m4a`（wav/mp3/m4a 均可）。
2. 放进 `video/voice_real/`。
3. 重新生成（画面会按真人录音的节奏自动重新对齐）：

```bash
python3 tts.py && python3 build.py && node render.js && node render.js --cover
```

只录了部分句子也可以：缺的句子会继续用 AI 占位音色（这时需要加 `--model-dir`，见下）。

## 从零生成（AI 占位配音）

依赖：Python 3（numpy、soundfile、sherpa-onnx）、Node 18+、Playwright + Chromium、ffmpeg。

```bash
pip install sherpa-onnx soundfile numpy
npm install
# 下载离线中文 TTS 模型（约 350MB）
curl -L -o kokoro.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xjf kokoro.tar.bz2
python3 tts.py --model-dir kokoro-multi-lang-v1_0
python3 build.py
node render.js            # 出成片
node render.js --cover    # 出封面
node render.js --preview 6,12,30   # 只截几张静帧检查画面
```

## 改文案 / 改画面

- 口播文案、字幕、音效触发点：`script.json`（`sub` 是字幕和录音文本；`say` 只在 AI 读错字时用来"纠音"；`cues.at` 是该句内触发动画的位置，0~1 表示句子进度）
- 画面与动画：`scene.html`（每个元素的出场时间都挂在句子/cue 上，改了配音不用手动对时间）
- 节奏参数（句间停顿、转场停顿、片尾定格）：`build.py` 顶部常量
