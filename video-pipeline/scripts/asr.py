import sys, json, numpy as np, soundfile as sf, sherpa_onnx
M=sys.argv[1]; wav=sys.argv[2]; out=sys.argv[3]
rec=sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=f"{M}/sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx",tokens=f"{M}/sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt",num_threads=4)
cfg=sherpa_onnx.VadModelConfig(); cfg.silero_vad.model=f"{M}/silero_vad.onnx"; cfg.silero_vad.min_silence_duration=0.35; cfg.silero_vad.max_speech_duration=12; cfg.sample_rate=16000
vad=sherpa_onnx.VoiceActivityDetector(cfg,buffer_size_in_seconds=60)
x,sr=sf.read(wav,dtype='float32'); ws=cfg.silero_vad.window_size
res=[]
def drain():
    while not vad.empty():
        seg=vad.front; s=rec.create_stream(); s.accept_waveform(16000,seg.samples); rec.decode_stream(s)
        st=seg.start/16000; res.append({"s":round(st,2),"e":round(st+len(seg.samples)/16000,2),"t":s.result.text}); vad.pop()
for i in range(0,len(x),ws):
    vad.accept_waveform(x[i:i+ws]); drain()
vad.flush(); drain()
json.dump(res,open(out,"w"),ensure_ascii=False,indent=0)
with open(out.replace(".json",".txt"),"w") as f:
    for r in res: f.write(f"[{int(r['s']//60):02d}:{r['s']%60:05.2f}] {r['t']}\n")
print(len(res))
