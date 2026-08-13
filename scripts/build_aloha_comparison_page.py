#!/usr/bin/env python3
"""Build an offline page for the saved 3D ALOHA ACT/DOT comparison."""

import argparse
import html
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    base = args.comparison.parent
    output_dir = args.output.parent
    cards = []
    for seed, title in [(1000, "DOT 完成，ACT 未完成"), (1001, "ACT 完成，DOT 未完成")]:
        path = output_dir / f"seed{seed}-act-vs-dot.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        cards.append(f'<article class="video-card"><h2>初始条件 {seed}</h2><p>{title}</p><video controls preload="metadata" playsinline src="{html.escape(path.name)}"></video></article>')
    rows = []
    for i, row in enumerate(data["rows"]):
        seed = row["seed"]
        act_success = row["act_success_reward4"]
        dot_success = row["dot_success_reward4"]
        outcome = "两者完成" if act_success and dot_success else "仅 ACT 完成" if act_success else "仅 DOT 完成" if dot_success else "两者未完成"
        video_cell = "—"
        if i < 10:
            act = base / "act" / "videos" / "aloha_0" / f"eval_episode_{i}.mp4"
            dot = base / "dot" / "videos" / f"eval_episode_{i}.mp4"
            if not (act.is_file() and dot.is_file()):
                raise FileNotFoundError(f"Missing episode {i} videos")
            act_href = html.escape(os.path.relpath(act, output_dir))
            dot_href = html.escape(os.path.relpath(dot, output_dir))
            video_cell = f'<a href="{act_href}">ACT</a> · <a href="{dot_href}">DOT</a>'
        rows.append(f'<tr data-outcome="{html.escape(outcome)}"><td>{seed}</td><td>{"成功" if act_success else "未完成"} · 奖励 {row["act_max_reward"]:g}</td><td>{"成功" if dot_success else "未完成"} · 奖励 {row["dot_max_reward"]:g}</td><td>{outcome}</td><td>{video_cell}</td></tr>')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三维 ALOHA 方块转交 · ACT 与 DOT 对比</title>
<style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;background:#f3f5f4;color:#1b2926}}
body{{max-width:1180px;margin:0 auto;padding:24px}}a{{color:#155c72}}header{{background:#e1ede9;padding:24px;border-radius:16px}}
h1{{margin:0 0 10px;font-size:clamp(1.6rem,4vw,2.5rem)}}p{{line-height:1.55}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}.metric,.video-card{{background:white;border-radius:14px;padding:18px;box-shadow:0 2px 8px #102c2012}}.metric strong{{display:block;font-size:2rem}}.videos{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}}video{{width:100%;border-radius:8px;background:#14221f}}.video-card h2{{margin:0}}.video-card p{{margin:6px 0 14px}}
.table-wrap{{overflow-x:auto;background:white;border-radius:14px}}table{{border-collapse:collapse;width:100%;min-width:680px}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #dfe5e2}}th{{background:#e9f0ed}}tr[hidden]{{display:none}}.controls{{margin:24px 0 12px}}select{{font:inherit;padding:8px 10px;border-radius:8px}}small{{color:#43544e}}
</style></head><body>
<header><h1>三维方块转交：两个模型的实际对比</h1><p>同一个 ALOHA 三维仿真任务、相同的 30 个初始条件。方块从右臂交到左臂，回合最高奖励达到 4 记为成功。</p></header>
<div class="metrics"><div class="metric">ACT<strong>{data['act_successes']}/30</strong>完成</div><div class="metric">DOT<strong>{data['dot_successes']}/30</strong>完成</div><div class="metric">已独立重复<strong>2 次</strong>成功标记一致</div></div>
<p>两段并排视频展示两种不同的结果。视频上方标注模型、初始条件与任务奖励；较短的一侧冻结末帧以对齐时长，原始视频仍保留在下表链接中。</p>
<div class="videos">{''.join(cards)}</div>
<div class="controls"><label for="filter">筛选结果：</label><select id="filter"><option value="all">全部 30 项</option><option value="仅 DOT 完成">仅 DOT 完成</option><option value="仅 ACT 完成">仅 ACT 完成</option><option value="两者完成">两者完成</option><option value="两者未完成">两者未完成</option></select></div>
<div class="table-wrap"><table><thead><tr><th>初始条件</th><th>ACT</th><th>DOT</th><th>对比</th><th>原始视频</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><small>评测器只保存每组前 10 个回合的视频；全部 30 个回合的奖励和成功判定保存在 <a href="{html.escape(os.path.relpath(args.comparison, output_dir))}">原始对比数据</a>。本结果属于 ALOHA，不代表原 SO101 的 ACT/SmolVLA 表现。</small></p>
<script>const f=document.getElementById('filter');f.addEventListener('change',()=>{{document.querySelectorAll('tbody tr').forEach(r=>r.hidden=f.value!=='all'&&r.dataset.outcome!==f.value)}});</script>
</body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
