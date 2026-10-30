# person cutout (RGBA) from a still image with u2net_human_seg
import sys, numpy as np, cv2, onnxruntime as ort
model, src, out = sys.argv[1], sys.argv[2], sys.argv[3]; x0, y0, x1, y1 = map(int, sys.argv[4:8])
img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
H, W = img.shape[:2]
sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
x = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
x = ((x / x.max()) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
m = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
m = (m - m.min()) / (m.max() - m.min() + 1e-6)
m = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)
a = np.clip((m - 0.35) / 0.3, 0, 1)
a = cv2.GaussianBlur(a, (0, 0), 1.2)
rgba = np.dstack([img, (a * 255).astype(np.uint8)])
ys, xs = np.where(a > 0.05); rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
cv2.imwrite(out, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)); print(rgba.shape)
