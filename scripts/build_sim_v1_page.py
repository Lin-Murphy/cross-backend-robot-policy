"""Create a local, standalone results page using saved simulation evidence."""
import html
import json
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve()
summary = json.loads((out / 'run/summary.json').read_text())
budget = json.loads((out / 'run/budget.json').read_text())
mount_layout = 'sim-v1-scene-20260925' in budget['scene']
if not mount_layout:
    raise ValueError('sim-v1 page expects the zero-wrist base-back scene')
if budget['initial_wrist_roll_deg'] != 0 or summary['status'] != 'success':
    raise ValueError('sim-v1 fixed demo did not meet the declared zero-wrist successful run')
source = root / 'artifacts/s1-base-back-8cm-policy-pilot-20260925'
analysis = json.loads((source / 'analysis.json').read_text())
shutil.copy2(source / 'analysis.json', out / 'model-check.json')
cards = []
for name, label in [('act', 'ACT'), ('smolvla', 'SmolVLA')]:
    trial = source / name
    dest = out / name
    dest.mkdir(exist_ok=True)
    for filename in ('summary.json', 'dual-camera.mp4'):
        shutil.copy2(trial / filename, dest / filename)
    item = analysis['models'][name]
    finding = f"20秒超时，接受{item['selected_targets']}个目标；夹爪参考点距胶带中心最近约{item['min_gripperframe_to_tape_center_m'] * 100:.1f}厘米。"
    cards.append(f'<section><h2>{label}：当前底座位置的开发试跑</h2><p>{html.escape(finding)}夹爪与胶带接触0次，任务未成功。模型在本次底座位置运行，但物理坐标仍未与实物核对。</p><video controls preload="metadata" src="{name}/dual-camera.mp4"></video><p><a href="{name}/summary.json">原始结果</a></p></section>')
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SO101 纯仿真演示</title><style>body{font:18px/1.6 system-ui;background:#101821;color:#e5edf4;max-width:1100px;margin:40px auto;padding:0 24px}section{background:#1d2a37;padding:24px;border-radius:14px;margin:24px 0}video{width:100%;background:black}a{color:#87d6ff}small{color:#b5c4d2}</style><h1>SO101 · 抓取与放置</h1><p>第一版：本地纯仿真。透明胶带卷 → 绿色垫子。</p><section><h2>固定脚本演示：成功</h2><p>本次重新运行 MuJoCo 物理仿真；腕部关节从0°开始，夹爪安装朝向偏转−90°；底座沿远离物品方向退后8厘米，camera2装在夹爪上方并随动。固定轨迹控制，不是 ACT 或 SmolVLA 的成功案例。开场camera2未拍到胶带，接近抓取时可见；相机安装仍是仿真假设。</p><video controls preload="metadata" src="run/dual-camera.mp4"></video><p><a href="run/summary.json">本次结果</a> · <a href="run/events.jsonl">完整状态与接触记录</a></p></section>CARDS<p><a href="model-check.json">两模型检查数值</a></p><small>仅本地展示，无机器人控制、训练、下载或远程服务。相机与物理参数使用开发假设，不声明实物一致性。两模型视频为当前底座位置已保存的开发试跑。</small></html>'''
(out / 'index.html').write_text(page.replace('CARDS', ''.join(cards)))
