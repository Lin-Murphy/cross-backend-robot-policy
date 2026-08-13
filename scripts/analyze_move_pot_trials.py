"""Analyze the frozen 40-slot record package without altering any trial records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from cross_backend.trial_analysis import analyze_trials
from scripts.move_pot_trial_records import validate


def render(result):
    def percent(v):return '未定义' if v is None else f'{100*v:.1f}%'
    def difference(v):return '未定义' if v is None else f'{100*v:+.1f} 个百分点'
    title='合成测试数据：不是实验结果' if result['synthetic_fixture'] else 'Move pot 已记录试验分析'
    lines=[f'# {title}','',
           '所有已启动试验计入分母，未执行单列；进行中的成功率为暂定值。成功计数为人工录入标签，并非工具证明任务成功。真实配对仅纳入文件完整性与冻结身份检查通过的记录；合成配对仅用于计算测试。','',
           '| 模型/条件 | 计划 | 已启动 | 未执行 | 进行中 | 成功 | 成功/已启动 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,g in result['groups'].items():
        lines.append(f"| {name} | {g['planned']} | {g['started']} | {g['not_run']} | {g['running']} | {g['confirmed_successes']} | {percent(g['success_rate_among_started'])} |")
    lines+=['','| 配对比较 | 差值方向 | 可分析配对数 | 成功比例差 |','|---|---|---:|---:|']
    for c in result['paired_comparisons']:
        lines.append(f"| {c['comparison']} | {c['direction']} | {c['complete_recorded_pairs']} | {difference(c['paired_success_difference'])} |")
    lines+=['','失败类别、终止耗时、成功完成耗时、逐初态差值与排除原因见summary.json。无结果时不输出假0%或补造完成时间。',
            f"真实证据核验通过trial数：{len(result['verified_trial_ids'])}；核验范围仅限文件完整性与冻结身份，不替代人工判定。",
            '这是描述性统计，不是显著性结论；成功耗时属于成功子集，不能忽略失败率独立比较快慢。','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    structural=validate(args.records)
    source=args.records/'trials.json';before=hashlib.sha256(source.read_bytes()).hexdigest()
    result=analyze_trials(json.loads(source.read_text()), verified_trial_ids=structural['verified_trial_ids'])
    result['record_validation']=structural;result['source_sha256']=before
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (args.output/'report.md').write_text(render(result))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print('Analysis saved without modifying trial records:',args.output)
