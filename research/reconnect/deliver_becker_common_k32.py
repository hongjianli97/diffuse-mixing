"""Local report and verified selected recovery package for common-K32 diagnostics."""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile
from audit_route_decision import ROOT, read, save, digest
OUT=ROOT/'runs/reconnect-becker-common-k32'
PARENT=ROOT/'runs/reconnect-stage3-evidence'
BASE='LTN_becker_common_k32'
DATE='20261107'
FILES=['becker_common_k32_core.py','becker_common_k32_study.py','becker_common_k32_protocol.json',
       'audit_becker_common_k32.py','deliver_becker_common_k32.py','BECKER_COMMON_K32_COMPARISON.md']

def document():
    r,a,ex=read(OUT/'results.json'),read(OUT/'audit.json'),read(OUT/'execution.json')
    assert a['passed'] and ex['completed']
    for name in ('main','audit'):
        shutil.copyfile('/tmp/ltn-becker-common-k32-'+name+'.log',OUT/(name+'.log'))
    passed=r['common_K32_spatial_diagnostic_resolved']
    title='共同K32三档空间诊断已分辨' if passed else '共同K32空间诊断失败或未决'
    scope=('只读定义一个最小独立层厚度δ对照：从原Becker场和已保存物质标签/质量平均定义出发，'
           '明确变化δ时哪些物质标签、初值、边界和物理量可固定、哪些须重建；冻结参考/时间/空间误差计账和资源预算，'
           '不预设δ变薄后误差必须降低，不因缺失H后移分辨而改原阈值。最多提出一个新增δ值，不运行新路径或参考PDE。'
           if passed else '只读分解本轮实际未决项目及其正项预算，保留全部结果；不自动补时间、空间、参考或δ计算。')
    save(OUT/'next-gate-spec.json',{'type':'read_only_minimal_independent_delta_control_definition' if passed else 'read_only_common_K32_unresolved_budget_review',
        'already_executed':False,'scope':scope,
        'sources':{p:digest(ROOT/p) for p in ['runs/reconnect-becker-common-k32/results.json','runs/reconnect-becker-common-k32/audit.json',
            'research/reconnect/BECKER_COMMON_K32_DIAGNOSTIC_CONTRACT.md','docs/METHOD.md','docs/CODEX_TASK.md','research/rebuild/becker.py']},
        'new_graphs':0,'new_candidate_paths':0,'new_reference_PDE_solves':0,'new_density_sources':0,'new_software_checks':0,
        'recorded_checks_total':502,'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,
        'stop':'no new delta data/graphs/solves, no full matrix, no H waiver or acceptance rewrite; no clipping,renormalization,state extrapolation or F/C fit',
        'new_full_Stage3_pass_allowed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-07','timezone':'Etc/UTC','decision':r['comparison_decision'],
        'common_K32_spatial_diagnostic_resolved':passed,'shifted_H_time_controls_resolved':r['shifted_H_time_controls_resolved'],
        'old_common_K16_spatial_passed':False,'old_K4_8_16_time_controls_preserved':True,
        'new_graphs':64,'new_actual_paths':2,'new_complete_independent_paths':2,'new_reference_PDE_solves':0,
        'new_density_sources':0,'new_software_checks':0,'recorded_checks_total':502,'next_gate_executed':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    obs='\n'.join(f"| {n} | {v['errors']['native']:.10g} | {v['errors']['common']:.10g} | {v['errors']['native_H']:.10g} | {v['errors']['common_H']:.10g} |" for n,v in r['by_grid'].items())
    gains='\n'.join(f"| {p['coarse_columns']}→{p['fine_columns']} | {k} | {p['gains'][k]:.10g} | {p['fixed_indicators'][k]:.10g} | {v} |" for p in r['adjacent_spatial_comparisons'] for k,v in p['decisions'].items())
    temporal='\n'.join(f"| {n} | {v['state_time_control']['decrease']:.10g} | {v['state_time_control']['fixed_budget']:.10g} | {v['state_time_control']['resolved']} | {v['shifted_H_time_control']['decrease']:.10g} | {v['shifted_H_time_control']['current_original_H_indicator']:.10g} | {v['shifted_H_time_control']['resolved']} |" for n,v in r['by_grid'].items())
    raw='\n'.join(f"- {n}列：质量变化{v['native_K32_mass_change']:.8g}，全部步极值[{v['min_all_steps']:.10g},{v['max_all_steps']:.10g}]；原始守卫{v['raw_all_steps_guards_passed']}，完整独立路径{v['all_complete_sequence_controls_passed']}。" for n,v in r['by_grid'].items())
    report=f'''# 共同K32三档空间诊断

2026-11-07（UTC）。**{title}**，决定 `{r['comparison_decision']}`。执行[冻结契约](BECKER_COMMON_K32_DIAGNOSTIC_CONTRACT.md)，Becker δ=.08、D=.02、T=.1、32/64/128列×8行，全部终态共同K32。仅32/64两条路径为新计算；128/K32、原K4/8/16路径及三个对齐参考全部复用。

## 原始误差与空间判别

浓度分母仍为原32列初始质量波动范数σ32，状态和参考均为固定物质单元质量平均；原生与共同32列观测同时报告。

| 列数 | 原生浓度/σ32 | 共同浓度/σ32 | 原生H参考误差 | 共同H参考误差 |
|---|---:|---:|---:|---:|
{obs}

| 空间档 | 量 | 实测改善 | 固定正项指标 | 决定 |
|---|---|---:|---:|---|
{gains}

八项空间比较均分辨={r['all_eight_spatial_comparisons_resolved']}；全部状态时间下降分辨={r['all_state_time_controls_resolved']}；原始/独立序列/容量守卫通过={r['all_raw_sequence_capacity_guards_passed']}。这是同一K32的一维列细化诊断，固定8行；不声称二维联合收敛、一般收敛阶或认证PDE误差界。

## 时间检查保持各自范围

| 列数 | 状态差下降 | 状态正项指标 | 分辨 | H差下降 | 原H指标 | H分辨 |
|---|---:|---:|---|---:|---:|---|
{temporal}

这里“下降”是旧K8→16变化减去实测K16→32变化。原三个网格K4/8/16的状态和H时间资格仍保持通过；后移8/16/32的各项结果独立列出，不互相替换。128列所有新表数字直接复用封存结果；32/64数值为本轮实际计算。

已知128后移H仍未分辨，其固定旧增量低于原保留下限。未作小H豁免、删正项或改阈值；未决检查继续保留。即使本轮空间诊断分辨，也不能追认原共同K16空间门、完整H时间验收或Stage3通过。上轮同128列H参考误差略增的记录保留，不能把空间误差降低写成H时间精度改善。

## 实现、计账和原始守卫

复用原K32采样包装器、bounded_periodic_voronoi普通面阻力、explicit_duration_action和冻结中点指数作用/完整独立原始统一化；旧COUNTS未改。32列K16几何输入来自最初封存的reconnect-becker-point，64列来自空间实验；两档共有33帧均逐位一致。

新64张图各自完成全单元半平面/共享面/周期镜像排除的独立证明，全部封存后才读取m、c0、旧状态或参考标量。候选不接收真实内部物质面；外面零标量通量、y周期、质量平均初值不变。

新增每档B*=旧B+2e_K32+新横向桥；e含机器项、原始尾项及完整独立末态差。共同J*=原容量范数因子×B*+全部旧共同桥+新汇总舍入/容量末态界。128完整预算直接复用。有限时间项换为实测K16→32状态或H变化，其余旧正项全部保留；H仍用原h_indicator与严格大于分类。无裁剪、质量/状态/概率行重归一化、外推终态或拟合F/C。

{raw}

独立审计重建64个已保存步骤的每面阻力、精确率、最小尾截断、原始权重和两条完整独立序列；核验共享通量/对称作用、完整物质汇总及80/100位正项计账。没有第二次实际指数传播或新参考PDE。参考仍为经验空间/时间差；原CN监控守卫通过不等于无条件最大值原理。

主流程{ex['elapsed_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB；独立审计{a['elapsed_seconds']:.3f}秒。64新图、2实际/2独立路径，0旧重跑/新参考/密度/软件检查，累计502，旧502未重跑。782原文件、15附件、111旧正式清单及四组历史差异保持；原README/METHOD/CODEX_TASK不改，所有失败/规划修正可追溯。

## 下一项尚未执行

{scope} 上限120秒/1024MiB，0新图/路径/参考/密度/软件检查。完整Stage3的独立层厚度与同后端oracle/kernel/ledger来源比较仍缺，不自动扩展全矩阵，不新增DNS、T5/T6、动态ORC或旧v080恢复作为当前解析工作的前提。

Stage3 incomplete，fluid ORC关闭，κ未算。仅本地PDF、代码/精选结果ZIP和Git恢复点，不发布或推送。ZIP不是全工作区镜像，不含15上传附件、未取回v080或完整聊天；完整历史保全仍需原工作区。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/audit_becker_common_k32.py \\
  --directory runs/reconnect-becker-common-k32
```
'''
    with (ROOT/'research/reconnect/BECKER_COMMON_K32_COMPARISON.md').open('x') as f:f.write(report)
    fmt=lambda x:f'{x:.6g}'
    obs_tex='\n'.join(f"{n}&{fmt(v['errors']['native'])}&{fmt(v['errors']['common'])}&{fmt(v['errors']['native_H'])}&{fmt(v['errors']['common_H'])}\\\\" for n,v in r['by_grid'].items())
    gain_tex='\n'.join(f"{p['coarse_columns']}到{p['fine_columns']}&{k.replace('_',' ')}&{fmt(p['gains'][k])}&{fmt(p['fixed_indicators'][k])}&{'分辨' if v=='fine_better_resolvably' else '未决/失败'}\\\\" for p in r['adjacent_spatial_comparisons'] for k,v in p['decisions'].items())
    time_tex='\n'.join(f"{n}&{fmt(v['state_time_control']['decrease'])}&{fmt(v['state_time_control']['fixed_budget'])}&{fmt(v['shifted_H_time_control']['decrease'])}&{fmt(v['shifted_H_time_control']['current_original_H_indicator'])}&{'分辨' if v['shifted_H_time_control']['resolved'] else '未分辨'}\\\\" for n,v in r['by_grid'].items())
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07 UTC}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 共同K32三档空间诊断}\\[5pt]{\large __TITLE__}\end{center}
\section*{1. 同一时间分辨率，仅新增两条路径}
Becker $\delta=.08,D=.02,T=.1$，32/64/128列、8行，全部共同K32。
只新增32/K32、64/K32各一条实际与完整独立路径，复用128/K32、全部旧K4/8/16和三个对齐参考。
没有旧路径重跑、新参考PDE或密度来源。状态、初值和参考均为原固定物质单元质量平均。
\section*{2. 原始终态误差}
浓度分母仍为原32列初始质量波动范数$\sigma_{32}$，原生/共同32列观测同时报告。
\begin{center}\begin{tabular}{rrrrr}\toprule
列数&原生浓度/$\sigma_{32}$&共同浓度/$\sigma_{32}$&原生H误差&共同H误差\\\midrule
__OBS__
\bottomrule\end{tabular}\end{center}
\section*{3. 八项实测空间判别}
改善定义为同K32粗网格参考误差减去细网格参考误差。预算包含两档各自实测K16到32变化及全部旧正项和新数值/观测桥；严格大于规则不改。
\begin{center}\begin{tabular}{llrrl}\toprule
空间档&量&实测改善&固定正项指标&结果\\\midrule
__GAINS__
\bottomrule\end{tabular}\end{center}
八项空间改善均分辨：__SPATIAL__；全部状态时间下降分辨：__STATE__。
原始质量/极值/方差/耗散、完整独立序列与容量守卫通过：__GUARDS__。

原共同K16空间决定继续保留未决。本轮是新的共同K32空间诊断，固定8行的一维列细化，
不声称二维联合收敛、一般收敛阶或认证连续PDE误差界。
\textbf{完整Stage3仍未通过，fluid ORC关闭，$\kappa$未计算。}
\newpage
\section*{4. 时间与H检查逐项保留}
下降是旧K8到16变化减去新K16到32变化；128列直接复用封存结果。
\begin{center}\small\begin{tabular}{rrrrrl}\toprule
列数&状态差下降&状态指标&H差下降&原H指标&H结果\\\midrule
__TIME__
\bottomrule\end{tabular}\end{center}
原三档K4/8/16状态与H时间资格保持通过。后移H的未分辨结果不能用旧资格或本轮空间改善替换。
128旧H增量小于保留下限，继续保留其未决。上轮同128列H参考误差略增的记录保持，不将空间改善写成H时间精度改善。
没有小H豁免、删正项或改阈值；本轮诊断不等于完整H时间验收。
\section*{5. 图先封存，全部正项不变}
复用原K32采样、普通面阻力、有界周期几何与显式时长指数传播/原始统一化。
旧COUNTS未改，两档各有33个与K16共有帧逐位一致。
64个新图先独立核验全部单元、共享面与周期镜像排除，再封存；其后才读取m/c0/旧状态/参考。
候选不接收内部真实物质面，外面零通量、y周期保持。

新增各档$B^*=B_{\rm old}+2e_{K32}+\text{新横向桥}$，其中e含机器、尾项及完整独立末态差。
共同预算保留容量算子范数及全部旧桥，再加新汇总舍入和容量末态界；128预算完全复用。
有限时间项用实测K16到32变化，H沿用原非线性指标。无裁剪、重归一化、状态外推或F/C拟合。
参考CN监控守卫通过不等于无条件最大值原理，经验差仍不是认证误差界。
\section*{6. 独立验证与下一项}
64步精确面阻力/率/最小尾截断/原始权重与两条完整独立序列、共享通量作用、完整物质汇总和80/100位预算审计通过。
主流程__SEC__秒/__RSS__MiB，审计__AUDIT__秒；无第二次实际指数传播或新参考PDE。
64新图、2实际/2独立路径、0旧重跑/新参考/密度/软件检查，累计502。782原文件、15附件、111旧正式清单及历史修正保留。

__NEXT__ 上限120秒/1024MiB，0新物理计算。独立层厚度和同后端三来源仍缺，不自动全矩阵扫描。

PDF、代码/精选结果ZIP和Git恢复点仅本地保存，不发布/推送。ZIP不是全工作区，不含15附件、未取回v080或完整聊天。
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_becker_common_k32.py \
--directory runs/reconnect-becker-common-k32
\end{verbatim}
\end{document}
'''
    mapping={'TITLE':title,'OBS':obs_tex,'GAINS':gain_tex,'TIME':time_tex,'SPATIAL':str(r['all_eight_spatial_comparisons_resolved']),
        'STATE':str(r['all_state_time_controls_resolved']),'GUARDS':str(r['all_raw_sequence_capacity_guards_passed']),
        'SEC':f"{ex['elapsed_seconds']:.3f}",'RSS':f"{ex['peak_RSS_MiB']:.2f}",'AUDIT':f"{a['elapsed_seconds']:.3f}",'NEXT':scope}
    for k,v in mapping.items():tex=tex.replace('__'+k+'__',v)
    tex=re.sub(r'(?<![0-9A-Za-z_.])(-?\d+(?:\.\d+)?)[eE]([+-]\d+)',lambda m:'\\ensuremath{'+m[1]+'\\times10^{'+str(int(m[2]))+'}}',tex)
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n112. [共同K32三档空间诊断](../research/reconnect/BECKER_COMMON_K32_COMPARISON.md)（2026-11-07 UTC）。\n\n'
        f'{title}，决定{r["comparison_decision"]}。仅32/64两条新路径，128/K32、旧K4/8/16和全部参考复用。64新图先审计封存，2实际/2完整独立路径，0旧重跑/新参考/密度/软件检查，累计502。'
        f'八项空间分辨={r["all_eight_spatial_comparisons_resolved"]}；后移H结果={r["shifted_H_time_controls_resolved"]}。全部旧正项/H公式/阈值及原K16空间未决保留，不自动升级完整Stage3。'
        '保存序列、完整物质观测和80/100位预算独立审计通过；782原文件、15附件、111旧正式清单及历史修正保持。'
        f'下一项尚未执行：{scope} 仅本地PDF/精选ZIP与Git恢复点；Stage3 incomplete，fluid ORC关闭，不发布/推送。\n')


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
    with zipfile.ZipFile(exports / 'LTN_stage3_evidence_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/becker_common_k32_protocol.json')['source_files'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    note = """LTN commonK32 three-column spatial diagnostic,2026-11-07 UTC.
+Only32/K32 and64/K32 are new:64 independently audited/sealed graphs,2 actual/2 complete independent paths.
+Reused128/K32 and all old paths and3 aligned references;no new reference PDE or density;count502,no new software tests.
+All old positive budget terms,H formula and thresholds retained. Read actual results.json for all8 spatial and3 shiftedH outcomes.
+Original K16 spatial decision and all historical failures retained;new scoped diagnostic cannot automatically pass full Stage3.
+Stage3 incomplete,fluid ORC OFF. Next gate registered,not executed. Local only,no publication/push.
+782 originals,research/docs and selected history only;not full workspace,15 attachments,missingv080 or chat.
+Verify review-manifest.json and original-handoff-sha256.json. Saved independent sequence/observations/budget audit:
+OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_common_k32.py --directory runs/reconnect-becker-common-k32
+""".replace('\n+','\n')
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
        with tempfile.TemporaryDirectory(prefix='ltn-common-k32-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_becker_common_k32.py',
                '--directory', 'runs/reconnect-becker-common-k32'], cwd=directory,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=180)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 45, 'prior_formal_manifests_verified': 111,
        'attachments_verified_in_workspace': 15, 'new_graphs': 64, 'new_actual_paths': 2, 'new_complete_independent_paths': 2, 'new_reference_PDE_solves': 0,
        'new_software_checks': 0, 'recorded_checks_total': 502, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# 共同K32空间诊断下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。本轮共同K32诊断已执行，下一项见登记；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['document','package'])
    args=p.parse_args()
    document() if args.mode=='document' else package()
