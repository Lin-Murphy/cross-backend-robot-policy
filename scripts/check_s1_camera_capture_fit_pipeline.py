"""Synthetic file handoff test for checkerboard capture -> offline fit; no devices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = out/'synthetic_inputs';raw.mkdir()
    collected = out/'accepted';collected.mkdir()
    captures = out/'capture_runs';captures.mkdir()
    pattern = np.zeros((7*48, 10*48), np.uint8)
    for row in range(7):
        for col in range(10):
            pattern[row*48:(row+1)*48, col*48:(col+1)*48] = 245 if (row+col)%2 else 15
    k = np.array([[610.,0,320.],[0,605.,240.],[0,0,1.]])
    src = np.array([[0,0],[480,0],[480,336],[0,336]],np.float32)
    xy = np.array([[-.09,-.063],[.09,-.063],[.09,.063],[-.09,.063]],np.float32)
    object_corners = np.column_stack([xy,np.zeros(4)]).astype(np.float32)
    rng = np.random.default_rng(20260925)
    rows = []
    try:
        for i in range(20):
            rvec = rng.uniform(-.45,.45,3)
            tvec = np.array([rng.uniform(-.04,.04),rng.uniform(-.04,.04),rng.uniform(.43,.60)])
            projected,_ = cv2.projectPoints(object_corners,rvec,tvec,k,np.zeros(5))
            dst = projected.reshape(4,2).astype(np.float32)
            transform = cv2.getPerspectiveTransform(src,dst)
            image = cv2.warpPerspective(pattern,transform,(640,480),flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,borderValue=180)
            source = raw/f'view-{i:03d}.png'
            if not cv2.imwrite(str(source),image):raise RuntimeError('Synthetic image write failed')
            capture = captures/f'view-{i:03d}'
            process = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/capture_s1_checkerboard_readonly.py'),
                                      '--offline-image',str(source),'--camera','follower',
                                      '--output',str(capture),'--views','1','--min-views','1'],
                                     capture_output=True,text=True)
            metadata = json.loads((capture/'capture.json').read_text())
            if process.returncode != 0 or metadata['accepted_views'] != 1:
                raise RuntimeError(f'Capture rejected synthetic view {i}: {process.stdout} {process.stderr}')
            position=metadata['records'][0].get('corner_position')
            if not position or len(position['bbox_px']) != 4 or not all(
                    0 <= v <= 1 for v in position['center_norm']) or not 0 < position['bbox_area_fraction'] < 1:
                raise RuntimeError(f'Capture did not record usable corner coverage for view {i}')
            photo = capture/'accepted/view-000.png'
            target = collected/f'view-{i:03d}.png'
            shutil.copy2(photo,target)
            rows.append({'view':i,'image_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                         'capture_status':metadata['status']})
        manifest=out/'synthetic-capture-manifest.json'
        manifest.write_text(json.dumps({'camera':'follower','status':'candidate_capture_complete',
                                        'accepted_views':len(rows),'records':[
                                            {'accepted':True,'path':str((collected/f'view-{r["view"]:03d}.png').resolve()),
                                             'sha256':r['image_sha256']} for r in rows]},indent=2)+'\n')
        fit = out/'fit'
        process = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/calibrate_s1_camera_offline.py'),
                                  '--images',str(collected),'--output',str(fit),
                                  '--camera','follower','--square-mm','18','--capture-manifest',str(manifest)],
                                 capture_output=True,text=True)
        result = json.loads((fit/'result.json').read_text())
        if process.returncode != 0 or result['status'] != 'candidate_not_accepted':
            raise RuntimeError(f'Offline fit failed: {process.stdout} {process.stderr} {result}')
        if len(result['views']) != 20 or sum(v['split']=='holdout' for v in result['views']) != 4:
            raise RuntimeError('Fit/holdout split mismatch')
        if result.get('capture_manifest_sha256') != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise RuntimeError('Fit did not preserve capture manifest identity')
        tampered=out/'tampered_inputs';shutil.copytree(collected,tampered)
        changed=tampered/'view-000.png';changed.write_bytes(changed.read_bytes()+b'tampered')
        rejected=out/'fit-tampered'
        negative=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/calibrate_s1_camera_offline.py'),
                                 '--images',str(tampered),'--output',str(rejected),
                                 '--camera','follower','--square-mm','18','--capture-manifest',str(manifest)],
                                capture_output=True,text=True)
        negative_result=json.loads((rejected/'result.json').read_text())
        if negative.returncode != 2 or 'capture manifest image hash mismatch' not in negative_result.get('reason',''):
            raise RuntimeError('Tampered capture image was not rejected')
        status = 'passed_synthetic_pipeline'
        error = None
    except Exception as exc:
        status = 'failed_synthetic_pipeline';error = f'{type(exc).__name__}: {exc}'
    summary = {'status':status,'synthetic_fixture':True,'hardware_access':False,
               'real_camera_calibrated':False,'generated_views':len(rows),
               'views':rows,'fit_status':result['status'] if 'result' in locals() else None,
               'fit_rms_px':result.get('fit_rms_px') if 'result' in locals() else None,
               'tampered_image_rejected':negative.returncode==2 if 'negative' in locals() else None,
               'error':error}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='views'}))
    return 0 if status=='passed_synthetic_pipeline' else 2


if __name__=='__main__':raise SystemExit(main())
