"""Read original orange videos; preserve fresh evidence without changing inputs."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path('/home/murphy/.cache/huggingface/lerobot/local/move_orange_20260919_194524')
OUT = Path('artifacts/r0-video-reconfirm-20260922')
OUT.mkdir(parents=True, exist_ok=False)
video = next(ROOT.glob('videos/**/*.mp4'))
episodes = [r for p in sorted(ROOT.glob('meta/episodes/**/*.parquet')) for r in pq.read_table(p).to_pylist()]
key = 'videos/observation.images.follower/'
targets = {}
for ep in episodes:
    start, end = ep[key+'from_timestamp'], ep[key+'to_timestamp']
    for fraction in (0.1, 0.5, 0.9):
        targets[round((start+(end-start)*fraction)*30)] = (ep['episode_index'], fraction)
records, shown = [], []
count = black = zero = 0
with av.open(str(video)) as container:
    for idx, frame in enumerate(container.decode(video=0)):
        a = frame.to_ndarray(format='rgb24')
        maximum = int(a.max())
        count += 1
        black += maximum <= 10
        zero += maximum == 0
        if idx in targets:
            ep, fraction = targets[idx]
            record = dict(frame=idx, episode=ep, fraction=fraction, pts_seconds=float(frame.time), minimum=int(a.min()), maximum=maximum, mean=float(a.mean()))
            records.append(record)
            if ep in (0, 24, 49):
                filename = f'episode-{ep:02d}-{fraction}.png'
                Image.fromarray(a).save(OUT/filename)
                shown.append((filename, a.copy(), record))
sheet = Image.new('RGB', (960, 810), 'white')
draw = ImageDraw.Draw(sheet)
for i, (filename, a, record) in enumerate(shown):
    x,y = (i%3)*320, (i//3)*270
    sheet.paste(Image.fromarray(a).resize((320,240)), (x,y+30))
    draw.text((x+4,y+4), f"ep {record['episode']} {record['fraction']:.0%} max={record['maximum']}", fill='black')
sheet.save(OUT/'contact-sheet.jpg')
independent = []
for index in (300, 15000, 29500):
    cmd = ['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{index})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    r = subprocess.run(cmd, capture_output=True, check=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8).reshape(480,640,3)
    Image.fromarray(a).save(OUT/f'ffmpeg-frame-{index}.png')
    independent.append(dict(frame=index, maximum=int(a.max()), mean=float(a.mean()), stderr=r.stderr.decode()))
configs = []
for p in sorted(Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_orange_v7/checkpoints').glob('*/pretrained_model/train_config.json')):
    d = json.loads(p.read_text())
    configs.append(dict(path=str(p), dataset=d['dataset'], sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), video=str(video), video_bytes=video.stat().st_size, video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(), episodes=len(episodes), decoded_frames=count, near_black_frames=black, zero_frames=zero, samples=records, ffmpeg_samples=independent, training_configs=configs)
(OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
(OUT/'run.log').write_text(f"Read-only source check completed {result['checked_utc']}\nPyAV decoded {count} frames; {black} near black; {zero} zero.\nFFmpeg independently decoded three selected frames.\nOriginal dataset and checkpoints unchanged.\n")
print(json.dumps({k:v for k,v in result.items() if k not in ('samples','training_configs')}, indent=2))
