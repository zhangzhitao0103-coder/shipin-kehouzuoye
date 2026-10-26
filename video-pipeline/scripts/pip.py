# teacher picture-in-picture: segment person per frame, track head, output circular RGBA PNG frames
import sys, os, subprocess, numpy as np, cv2, onnxruntime as ort
model, video, t0, t1, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
FPS=25; OUT=480; X0,Y0,W,H=880,0,1040,760   # search region (right-top of 1080p frame)
os.makedirs(outdir, exist_ok=True)
sess=ort.InferenceSession(model, providers=['CPUExecutionProvider']); inp=sess.get_inputs()[0].name
cmd=['ffmpeg','-v','error','-ss',str(t0),'-t',str(t1-t0),'-i',video,'-vf',f'fps={FPS},crop={W}:{H}:{X0}:{Y0}','-f','rawvideo','-pix_fmt','rgb24','-']
p=subprocess.Popen(cmd,stdout=subprocess.PIPE); frames=[]; masks=[]
mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
while True:
    b=p.stdout.read(W*H*3)
    if len(b)<W*H*3: break
    f=np.frombuffer(b,np.uint8).reshape(H,W,3)
    x=cv2.resize(f,(320,320),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    x=((x/ x.max())-mean)/std; x=x.transpose(2,0,1)[None].astype(np.float32)
    m=sess.run(None,{inp:x})[0][0,0]; m=(m-m.min())/(m.max()-m.min()+1e-6)
    m=cv2.resize(m,(W,H),interpolation=cv2.INTER_LINEAR)
    frames.append(f); masks.append(m)
# track head: top of person + horizontal centre of upper body
cs=[]
for m in masks:
    b=m>0.5; rows=np.where(b.sum(1)>12)[0]
    if len(rows)==0: cs.append(None); continue
    top=rows[0]; band=b[top:top+150]; xs=np.where(band.any(0))[0]
    cs.append((xs.mean() if len(xs) else W/2, top))
valid=[c for c in cs if c is not None]; med=np.median(np.array(valid),0)
cs=[c if c is not None else tuple(med) for c in cs]
cx=np.array([c[0] for c in cs]); ty=np.array([c[1] for c in cs])
k=13; ker=np.ones(k)/k
sm=lambda a: np.convolve(np.pad(a,(k//2,k//2),mode='edge'),ker,mode='valid')
cx,ty=sm(cx),sm(ty)
SIDE=360
yy,xx=np.mgrid[0:OUT,0:OUT]; r=np.hypot(xx-OUT/2+.5,yy-OUT/2+.5)
circ=np.clip(OUT/2-1-r,0,1)
bg=np.zeros((OUT,OUT,3),np.float32); g=(yy/OUT)[...,None]
bg[:]=(1-g)*np.array([255,236,228])+g*np.array([255,214,200])
for i,(f,m) in enumerate(zip(frames,masks)):
    x0=int(round(cx[i]-SIDE/2)); y0=int(round(ty[i]-SIDE*0.10))
    pad=SIDE; F=cv2.copyMakeBorder(f,pad,pad,pad,pad,cv2.BORDER_REPLICATE); Mm=cv2.copyMakeBorder(m,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
    cf=F[y0+pad:y0+pad+SIDE,x0+pad:x0+pad+SIDE]; cm=Mm[y0+pad:y0+pad+SIDE,x0+pad:x0+pad+SIDE]
    cf=cv2.resize(cf,(OUT,OUT),interpolation=cv2.INTER_AREA).astype(np.float32); cm=cv2.resize(cm,(OUT,OUT))[...,None]
    cm=np.clip((cm-0.3)/0.4,0,1)
    rgb=cf*cm+bg*(1-cm)
    out=np.dstack([rgb,circ*255]).astype(np.uint8)
    cv2.imwrite(f"{outdir}/p_{i:05d}.png",cv2.cvtColor(out,cv2.COLOR_RGBA2BGRA))
print(len(frames),"frames", "median head", med)
