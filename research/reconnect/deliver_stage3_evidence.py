"""Local PDF and selected recovery ZIP for the read-only Stage3 evidence review."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT, read, save, digest

OUT = ROOT / 'runs/reconnect-stage3-evidence'
PARENT = ROOT / 'runs/reconnect-becker-fine-time'
BASE = 'LTN_stage3_evidence'
DATE = '20261107'
FILES = ['stage3_evidence_review.py', 'audit_stage3_evidence.py',
         'stage3_evidence_protocol.json', 'deliver_stage3_evidence.py',
         'STAGE3_EVIDENCE_REVIEW.md', 'BECKER_COMMON_K32_DIAGNOSTIC_CONTRACT.md']


def document():
    assert read(OUT / 'audit.json')['passed']
    for name in ('main', 'audit'):
        shutil.copyfile('/tmp/ltn-stage3-evidence-' + name + '.log', OUT / (name + '.log'))
    contract = 'research/reconnect/BECKER_COMMON_K32_DIAGNOSTIC_CONTRACT.md'
    save(OUT / 'next-gate-spec.json', {
        'type': 'new_common_K32_three_column_spatial_diagnostic_with_all_H_outcomes_retained',
        'already_executed': False, 'contract': contract, 'contract_sha256': digest(ROOT / contract),
        'new_columns_K': [[32, 32], [64, 32]], 'reuse_columns_K': [[128, 32]],
        'delta': .08, 'D': .02, 'T': .1, 'rows': 8,
        'max_new_graphs': 64, 'max_new_candidate_paths': 2, 'max_new_complete_independent_paths': 2,
        'new_reference_PDE_solves': 0, 'new_density_sources': 0, 'new_software_checks': 0,
        'recorded_checks_total': 502, 'main_wall_seconds_cap': 900, 'main_peak_RSS_MiB_cap': 1024,
        'reuse_all_old_paths_and_three_aligned_references': True,
        'graph_audit_and_seal_before_scalar_reference_loading': True,
        'retain_original_K4_8_16_qualification_and_all_shifted_H_failures': True,
        'same_positive_budget_H_formula_sigma_and_strict_classification': True,
        'all_eight_new_spatial_decisions_must_be_reported': True,
        'full_Stage3_or_original_K16_pass_allowed': False,
        'stop': 'save failure/time-limit/unresolved; no automatic K64, new spatial/reference/delta/density sweep or acceptance rewrite',
        'stage3_physical_acceptance': 'incomplete', 'fluid_orc_enabled': False, 'kappa_computed': False})
    save(OUT / 'stage3-status.json', {
        'date': '2026-11-07', 'timezone': 'Etc/UTC', 'read_only_review_completed': True,
        'original_common_K16_spatial_passed': False, 'original_K4_8_16_time_passed': True,
        'mixed_time_concentration_diagnostic_resolved': True, 'shifted_H_time_resolved': False,
        'remaining_evidence': ['common-time Becker spatial diagnostic',
            'retained unresolved shifted H check', 'independent layer-thickness control',
            'same-backend oracle/kernel/ledger closure'],
        'new_graphs': 0, 'new_candidate_paths': 0, 'new_reference_PDE_solves': 0,
        'new_density_sources': 0, 'new_software_checks': 0, 'recorded_checks_total': 502,
        'next_gate_executed': False, 'stage3_physical_acceptance': 'incomplete',
        'fluid_orc_enabled': False, 'kappa_computed': False})
    tex = r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,tabularx,hyperref,fancyhdr}
\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07 UTC}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries Stage3证据与剩余缺口}\\[5pt]{\large 只读复核：原判据、旧结论保持}\end{center}
\textbf{Stage3尚未通过，fluid ORC关闭，$\kappa$未计算。}
本轮不新增图、状态、参考PDE、密度来源或软件测试，累计502；未取回旧v080/246不计入。
\section*{1. 原要求对应的已保存证据}
\begin{tabularx}{\textwidth}{p{25mm}XX}\toprule
原要求&已保存结果&适用范围或缺口\\\midrule
T1均匀扩散&三空间档周期正弦、质量平均、解析参考通过&只限登记案例，不是一般收敛阶证明\\
T2客观性/D=0&平移、非刚性D=0、精确共转边界通过&受输入和边界模式限定，$\kappa$未算\\
T3压缩/剪切&仿射压缩和剪切三档空间/时间及方向响应&原N32压缩未决保留，不宣称任意平滑流动准入\\
T4静态接触&独立接触面的三档oracle控制通过&不替代移动激波或三来源验收\\
T4移动Becker&原时间三元组通过；新增混合时间浓度诊断分辨&共同K16空间仍有两项未决；后移H未决\\
独立层厚度&原CODEX\_TASK第31行已有要求&最终点后端固定$\delta=.08$，未闭合\\
三密度来源&oracle隔离与旧密度理论可复用&同后端oracle/kernel/ledger未闭合\\\bottomrule
\end{tabularx}
\section*{2. 时间资格与空间结论分开}
原32/64/128列的K4/8/16状态与H下降均超过原正项指标。共同K16的空间判别：
32到64四项分辨；64到128两项H分辨、两项浓度未决，原空间决定仍为false。

新增64/K16对128/K32四项改善均超过固定指标。128列原生浓度误差/原$\sigma_{32}$由
$4.344644890\times10^{-5}$降至$3.269596647\times10^{-5}$。
这是混合时间诊断，不是共同K32三档空间实验，不能追认原门通过。
\section*{3. 后移H有可直接核验的下限障碍}
令$d=|H_8-H_{16}|=1.582723197\times10^{-8}$，
$d'=|H_{16}-H_{32}|=1.207153932\times10^{-8}$，原保留下限
$J=1.751753825\times10^{-7}$。则
\[d-d'\leq d<J\leq J^*=1.996568516\times10^{-7}.\]
即使这个检查的新增量为零，仍不能超过保留下限。论证只针对固定旧增量/预算，不预测K64。
同一128列H参考误差从$2.618241188\times10^{-6}$变为$2.630312728\times10^{-6}$，略增；
不能称H时间精度改善，也不据此宣称已分辨的物理退化。
\newpage
\section*{4. 不修改原规范或已登记检查}
METHOD第132行要求空间/时间各至少三档，分别变化；没有逐字要求每一组后移三元组无限持续可分辨下降。
但具体登记的H检查仍按原式保留未决，\textbf{不豁免、不删正项、不降低阈值}。
原K4/8/16通过与后移H未决可以同时存在，不能互相替换。

十组实现锁确认共享普通阻力/传播核心哈希相同；有界和滑移周期几何有各自扩展，不能等同整个边界API。
当前研究指数传播不等于原默认隐式Euler/生产CLI迁移。参考差、时间差仍是经验指标，
参考CN监控守卫通过不等于无条件最大值原理。
\section*{5. 下一项已登记，尚未执行}
共同K32的32/64/128列空间诊断：只补32/K32、64/K32两条实际与完整独立路径，
复用128/K32、全部旧路径与三个对齐参考。固定$\delta=.08,D=.02,T=.1$、8行、原m/c0、
普通阻力、$\sigma_{32}$和H公式。新图全部独立审计封存后才读标量/参考。

新32/64预算按已封存单档规则扩展：保留全部旧正项，加两倍新路径数值指标和新横向桥；
共同观测再加汇总舍入及容量缺陷界。有限时间项用实测K16到32变化，不能外推状态或归一化。
报告两组相邻空间的八项实测改善；新状态检查、原时间资格、后移H分类分栏列出。

最多64新图、2实际/2独立路径，0新参考/密度/软件测试，主计算900秒/1024MiB。
失败、超限或空间未决即停，不自动加K64、空间档、参考、$\delta$或密度扫描。
即使共同时间空间诊断分辨，也不自动通过完整Stage3，后移H未决继续公开。

它能回答的是：统一时间分辨率后，这个一维oracle空间改善能否超过原规则的不确定度。
随后仍需分别登记独立层厚度与同后端三来源验证；不能承诺通过日期。
不新增DNS、T5/T6、动态ORC或旧v080恢复作为当前解析试验前提。
\section*{6. 可复核交付与保全}
标准库只读复核64份封存来源、12条原文引用、10组状态；独立审计以80/100位复算正项比较和H下限。
主复核__SEC__秒/__RSS__MiB，无新物理计算；782原文件、15附件、110旧正式清单及四组历史差异保持。
旧PHASE与覆盖表按原日期保留，后续证据在新账本列出。

PDF、代码/精选结果ZIP与本地Git恢复点仅本地保存，不发布或推送。
ZIP不是全工作区镜像，不含15上传附件、未取回v080或完整聊天。
\begin{verbatim}
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_stage3_evidence.py \
--directory runs/reconnect-stage3-evidence
\end{verbatim}
\textbf{Stage3 incomplete；fluid ORC关闭。}
\end{document}
'''
    ex = read(OUT / 'execution.json')
    tex = tex.replace('__SEC__', f"{ex['elapsed_seconds']:.3f}").replace('__RSS__', f"{ex['peak_RSS_MiB']:.2f}")
    with (OUT / f'{BASE}_report_{DATE}.tex').open('x') as f:
        f.write(tex)
    with (ROOT / 'docs/RESEARCH_STATUS.md').open('a') as f:
        f.write('\n\n111. [Stage3证据与H障碍只读复核](../research/reconnect/STAGE3_EVIDENCE_REVIEW.md)（2026-11-07 UTC）。\n\n'
            '64份封存来源、12条原文和10组案例状态核对，80/100位正项比较独立复算；原K16空间6项分辨/2项未决、旧K4/8/16时间通过、混合K16/K32四项分辨及后移H未决均保持。'
            '同后端独立层厚度与三密度来源仍未闭合。0新图/路径/参考/密度/软件测试，累计502；782原文件、15附件、110旧正式清单及四组历史差异保持。'
            '下一项已冻结共同K32三档空间诊断，只补32/64两条路径，复用128/K32和全部参考；本轮未执行，不豁免后移H、不追认旧空间门。'
            '仅本地PDF/精选ZIP与Git恢复点；Stage3 incomplete，fluid ORC关闭，不发布/推送。\n')


def package():
    exports = ROOT / 'exports'
    pdf = exports / f'{BASE}_report_{DATE}.pdf'
    archive = exports / f'{BASE}_code_results_{DATE}.zip'
    seal = OUT / 'sha256.json'
    assert not any(p.exists() for p in (pdf, archive, seal))
    assert read(OUT / 'audit.json')['passed']
    assert read(OUT / 'PDF-visual-review.json')['all_pages_reviewed']
    info = subprocess.check_output(['pdfinfo', str(OUT / pdf.name)], text=True)
    assert 'Pages:           2' in info
    log = (OUT / f'{BASE}_report_{DATE}.log').read_text()
    assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT / pdf.name, pdf)
    save(OUT / 'PDF-generation.json', {'pages': 2, 'sha256': digest(pdf),
         'no_overfull_or_missing_characters': True, 'all_pages_visually_reviewed': True})
    entries = read(PARENT / 'sha256.json')
    for p, h in entries.items():
        assert digest(ROOT / p) == h, p
    entries[str((PARENT / 'sha256.json').relative_to(ROOT))] = digest(PARENT / 'sha256.json')
    for name in FILES:
        entries['research/reconnect/' + name] = digest(ROOT / 'research/reconnect' / name)
    for p in OUT.rglob('*'):
        if p.is_file():
            entries[str(p.relative_to(ROOT))] = digest(p)
    entries[str(pdf.relative_to(ROOT))] = digest(pdf)
    save(seal, dict(sorted(entries.items())))
    with zipfile.ZipFile(exports / 'LTN_becker_fine_time_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/stage3_evidence_protocol.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    note = '''LTN Stage3 read-only evidence review, 2026-11-07 UTC.
64 sealed sources/12 original quotes/10 case records, independent Decimal80/100 scalar budget audit.
No new graphs/states/PDE/density/tests; recorded checks502. Stage3 incomplete,fluid ORC OFF.
Original K16 spatial unresolved, old K4/8/16 time passed, mixed-time diagnostic resolved, shifted H unresolved.
Next commonK32 spatial diagnostic registered but NOT executed; no H waiver or retrospective acceptance.
782 originals,research/docs and selected results only; not full workspace,15 attachments,missingv080 or chat.
Verify review-manifest.json and original-handoff-sha256.json. Portable saved evidence audit:
PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_stage3_evidence.py --directory runs/reconnect-stage3-evidence
No publication or remote push. Full historical preservation needs original workspace.
'''
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in manifest:
            z.write(ROOT / p, p)
        z.writestr('review-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        z.writestr('original-handoff-sha256.json', json.dumps(originals, indent=2) + '\n')
        z.writestr('DELIVERY_README.txt', note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p, h in manifest.items():
            assert hashlib.sha256(z.read(p)).hexdigest() == h
        with tempfile.TemporaryDirectory(prefix='ltn-stage3-evidence-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_stage3_evidence.py',
                '--directory', 'runs/reconnect-stage3-evidence'], cwd=directory,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 64, 'prior_formal_manifests_verified': 110,
        'attachments_verified_in_workspace': 15, 'new_graphs_or_states_or_reference_solves': 0,
        'new_software_checks': 0, 'recorded_checks_total': 502, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# Stage3证据复核下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。下一项共同K32空间诊断尚未执行；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['document', 'package'])
    args = parser.parse_args()
    document() if args.mode == 'document' else package()
