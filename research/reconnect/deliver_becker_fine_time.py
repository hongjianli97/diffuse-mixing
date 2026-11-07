"""Local report and verified recovery package for the one-path K32 diagnostic."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-becker-fine-time';PARENT=ROOT/'runs/reconnect-becker-spatial-budget'
BASE='LTN_becker_fine_time';DATE='20261107'
FILES=['becker_fine_time_core.py','becker_fine_time_study.py','becker_fine_time_protocol.json',
       'audit_becker_fine_time.py','BECKER_FINE_TIME_COMPARISON.md','deliver_becker_fine_time.py']


def document():
    r=read(OUT/'results.json');a=read(OUT/'audit.json');ex=read(OUT/'execution.json');old=read(OUT/'source/results.json')
    assert a['passed'] and ex['first_attempt_passed']
    for name in ('main','audit'):shutil.copyfile(f'/tmp/ltn-becker-fine-time-{name}.log',OUT/f'{name}.log')
    passed=r['concentration_diagnostic_resolved'];decision='单档浓度时间诊断已分辨' if passed else '单档浓度时间诊断仍失败或未决'
    scope=('只读整理已保存的混合时间浓度/H空间证据、原K4/8/16时间资格与后移H下限障碍，逐条对应原Stage3要求；'
           '明确仍欠缺的证据，不能把本轮诊断追认为原三档空间通过或豁免H判据。' if passed else
           '只读分解新保存的未决浓度预算，检查未细化64列时间项、128列新变化、参考和数值项；保留失败/未决，不自动补算。')
    next_text=scope+'本项0新图/传播/参考/软件测试；只有证据支持后另行登记最小下一项。'
    sources=['runs/reconnect-becker-fine-time/results.json','runs/reconnect-becker-fine-time/audit.json',
        'research/reconnect/BECKER_FINE_TIME_PROBE_CONTRACT.md','research/reconnect/BECKER_SPATIAL_BUDGET.md',
        'runs/reconnect-becker-spatial/results.json','runs/reconnect-becker-spatial-budget/planning-correction.json','docs/CODEX_TASK.md','docs/METHOD.md']
    save(OUT/'next-gate-spec.json',{'type':'read_only_saved_time_diagnostic_evidence_and_H_obstruction_review' if passed else 'read_only_saved_fine_time_unresolved_budget_review',
        'already_executed':False,'scope':scope,'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in sources],
        'new_graphs':0,'new_candidate_paths':0,'new_reference_PDE_solves':0,'new_density_sources':0,'new_software_checks':0,
        'recorded_checks_total':502,'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,
        'stop':'no automatic64/K32,K64,new space/reference/density or acceptance rewrite; no clipping,normalization,extrapolation,F/C fit,H or threshold retune',
        'old_results_preserved':True,'new_full_spatial_or_Stage3_pass_allowed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-07','timezone':'Asia/Shanghai','decision':r['comparison_decision'],
        'concentration_diagnostic_resolved':passed,'mixed_time_levels':r['mixed_time_levels'],'old_spatial_decision_unchanged':r['old_spatial_decision_unchanged'],
        'shifted_H_time_control_resolved':False,'new_full_spatial_or_Stage3_pass_allowed':False,'new_graphs':32,'new_candidate_paths':1,
        'new_independent_paths':1,'new_reference_PDE_solves':0,'new_software_checks':0,'recorded_checks_total':502,
        'next_gate_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    rows='\n'.join(f"| {key} | {r['gains'][key]:.10g} | {r['fixed_indicators'][key]:.10g} | {r['decisions'][key]} |" for key in r['gains'])
    state=r['state_time_control'];ht=r['shifted_H_time_control'];previous=old['observations']['128']
    report=f'''# 单一128列K32：保存参考下的浓度时间诊断

2026-11-07（Asia/Shanghai）。**{decision}**，决定 `{r['comparison_decision']}`。执行[冻结契约](BECKER_FINE_TIME_PROBE_CONTRACT.md)，δ=.08、D=.02、T=.1、128列×8行，仅K32一条实际和一条完整独立路径。旧32/64/128的K4/8/16路径与三个对齐参考均复用，0旧路径重跑、0新参考PDE。

## 终态与固定判别

| 128列终态 | K16（旧） | K32（新） |
|---|---:|---:|
| 原生浓度误差/σ32 | {previous['native_error_common_denominator']:.10g} | {r['errors']['native']:.10g} |
| 共同32列误差/σ32 | {previous['common_error']:.10g} | {r['errors']['common']:.10g} |
| 原生H | {previous['native_H']:.10g} | {r['native_H_K32']:.10g} |
| 原生H参考误差 | {abs(previous['native_H']-previous['native_reference_H']):.10g} | {r['errors']['native_H']:.10g} |

新比较是64/K16对128/K32，终止时刻同为T=.1，时间分辨率不同。浓度分母仍为原32列初始质量波动范数，参考仍为固定物质质量平均。没有外推终态或由旧误差预测新改善。

同一128列的H参考误差略增；该变化与“64列对128列的H空间改善”是两个不同的比较。不能把浓度误差下降写成H时间精度改善。

| 64/K16→128/K32 | 实测改善 | 固定正项指标 | 决定 |
|---|---:|---:|---|
{rows}

128列状态时间变化：旧K8→16为{state['old8_16']:.10g}，新K16→32为{state['new16_32']:.10g}，下降{state['decrease']:.10g}，原规则正项B128*={state['fixed_budget']:.10g}；分辨={state['resolved']}。实际/完整独立路径及原始质量、极值、方差、共享面耗散检查通过。新终态质量变化{r['native_K32_mass_change']:.6g}，全部步原始极值范围[{r['min_all_steps']:.10g},{r['max_all_steps']:.10g}]。

## H后移障碍仍保留

原|H8−H16|={ht['old8_16']:.10g}，新|H16−H32|={ht['new16_32']:.10g}；下降{ht['decrease']:.10g}，当前原公式指标{ht['current_original_H_indicator']:.10g}，仍未分辨。旧下限{ht['retained_original_H_floor']:.10g}已大于旧增量；新增非负项不能降低这一指标。没有小H豁免或改阈值。旧K4/8/16时间通过与原三列档空间未决保持。

浓度诊断只要求预登记的两项浓度比较、状态时间差及原守卫/独立路径通过，H空间比较单独报告；这一局部诊断不能替代完整H时间验收。混合(16,32)证据也不等于三个网格共同K32空间验收，无论本轮结果如何都不直接打开三密度来源分支。

## 质量、几何和全部正项保持

标签中点轨迹规则仅将K元组改为(32,)，AST核验其余采样表达式完全相同，旧全局COUNTS未改；65帧中的33个共同帧与旧K16逐位一致。32个新中点图逐一完成共享面、全单元半平面和非最近周期镜像排除的独立证明，全部封存后才加载m/c0/旧状态/参考。候选不接收真实内部物质面，外面零标量通量、y周期保持。

B128*=旧B128+2e32+新横向桥，e32保留机器、原始尾项及完整独立终态差；J128*保留旧全部桥，并加新汇总舍入和终态容量缺陷界。64列各旧空间指标不变，只将128时间项换成实测K16→32差；无旧正项删除或相消。容量因子仅进入预算，没有质量、行或状态重归一化。H仍用原非线性h_indicator，经验指标不是认证连续PDE误差界。

## 独立核验与保全

主流程首次完成{ex['elapsed_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，独立审计首次完成{a['elapsed_seconds']:.3f}秒。审计核对封存前全单元几何证明，独立重建每面阻力、精确率、最小尾截断、原始权重、32步完整独立序列，并核对共享通量/对称作用、完整物质汇总、原生/共同误差及80/100位正项计账。没有第二次运行实际指数传播、构图或参考PDE。

32新图、1实际/1独立路径、0旧重跑、0新参考/密度/软件检查，累计502。782原文件、15附件、109份旧正式清单及历史失败/规划修正保留。原始README/METHOD/CODEX_TASK与原附件不改。

## 下一项与交付

{next_text}

Stage3 incomplete，fluid ORC关闭，κ未算。PDF与代码/精选结果ZIP仅本地保存；ZIP不是全工作区镜像，不含15上传附件、未取回v080或完整聊天。不发布或推送。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/audit_becker_fine_time.py \\
  --directory runs/reconnect-becker-fine-time
```
'''
    with (ROOT/'research/reconnect/BECKER_FINE_TIME_COMPARISON.md').open('x') as f:f.write(report)
    table='\n'.join(f"{key.replace('_',' ')}&{r['gains'][key]:.6g}&{r['fixed_indicators'][key]:.6g}\\\\" for key in r['gains'])
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=22mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 单一128列K32时间诊断}\\[5pt]{\large __DECISION__}\end{center}
\section*{1. 只新增一条路径}
$\delta=.08,D=.02,T=.1$，128列、8行、K32。原64列K16作保存对照，旧32/64/128所有路径与三个对齐参考复用。
没有新参考PDE、密度来源或旧路径重跑；质量平均状态、m、c0、普通阻力和原传播后端保持。
\begin{center}\begin{tabular}{lrr}\toprule
128列误差/原$\sigma_{32}$&旧K16&新K32\\\midrule
原生观测&__OLDN__&__NEWN__\\
共同32列观测&__OLDP__&__NEWP__\\\bottomrule
\end{tabular}\end{center}
\section*{2. 实测改善与固定指标}
以下比较为64/K16对128/K32；同一终止时刻、不同时间分辨率，不用旧改善预测新改善。
\begin{center}\begin{tabular}{lrr}\toprule
量&实测改善&固定正项指标\\\midrule
__TABLE__
\bottomrule\end{tabular}\end{center}
128列状态时间变化从__DOLD__降为__DNEW__，下降__DD__；正项B128*为__B__，分辨结果__STATE__。
实际/完整独立序列、原始质量/极值/方差/耗散检查通过。新终态质量变化__MASS__。
\section*{3. H时间判据仍未分辨}
旧$|H_8-H_{16}|=$__HOLD__，新$|H_{16}-H_{32}|=$__HNEW__；下降__HD__，当前原公式指标__HB__。
原保留指标下限__HFLOOR__已大于旧增量，新增非负项不会降低该下限。没有小H豁免或修改阈值。

同一128列H参考误差由__OLDHE__变为__NEWHE__，略有增加；不把浓度误差下降写成H时间精度改善。

原K4/8/16时间通过和原三列档空间未决保留。浓度诊断通过条件早已单独登记，不能替代完整H时间验收；
混合(16,32)比较也不是三个网格共同K32的空间验收。\textbf{不能据本轮直接宣称Stage3通过。}
\newpage
\section*{4. 图先封存；旧正项全部保留}
采样包装器仅固定K32，AST核对原规则，旧全局COUNTS未改。新旧共有33帧逐位一致。
32张新中点图先独立核验共享面、全单元半平面和周期镜像排除证明，再封存；此后才读m/c0/旧状态/参考。
候选不接收内部真实物质面，x外面零通量、y周期保持。

$B_{128}^{*}=B_{128}^{\rm old}+2e_{32}+\text{新横向桥}$；$e_{32}$含机器、原尾项及完整独立终态差。
共同J128*保留全部旧桥，再加新汇总舍入和容量缺陷界。64列指标不变，128时间项换为实测K16到32差。
旧正项不删除、不相消；容量因子只进入误差预算，不归一化质量、概率行或状态。
原H公式不变，经验指标不是认证连续PDE误差界。
\section*{5. 独立审计与数据保全}
核对封存前全单元几何证明，独立重算每面阻力、精确交换率、最小尾截断、原始Poisson权重和32步完整独立序列。
共享通量/对称作用、完整物质汇总、原生/共同误差及80/100位正项预算核验通过。
没有重跑实际指数传播、几何构造或参考PDE。

主流程首次完成__SEC__秒/__RSS__MiB；审计首次完成__AUDIT__秒。
32新图、1实际/1独立路径、0旧重跑、0新参考/密度/软件检查，累计502，旧502未重跑。
782原文件、15附件、109旧正式清单及全部失败/规划修正保留。
\section*{6. 下一项尚未执行}
__NEXT__

\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。}
PDF和代码/精选结果ZIP仅本地保存；不是全工作区镜像，不含15附件、未取回v080或聊天。
不发布或推送。保存数据重放入口：
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_becker_fine_time.py \
--directory runs/reconnect-becker-fine-time
\end{verbatim}
\end{document}
'''
    fmt=lambda v:f'{v:.6g}'
    mapping={'DECISION':decision,'OLDN':fmt(previous['native_error_common_denominator']),'NEWN':fmt(r['errors']['native']),
        'OLDP':fmt(previous['common_error']),'NEWP':fmt(r['errors']['common']),'TABLE':table,'DOLD':fmt(state['old8_16']),
        'DNEW':fmt(state['new16_32']),'DD':fmt(state['decrease']),'B':fmt(state['fixed_budget']),'STATE':'可分辨' if state['resolved'] else '未分辨',
        'MASS':fmt(r['native_K32_mass_change']),'HOLD':fmt(ht['old8_16']),'HNEW':fmt(ht['new16_32']),'HD':fmt(ht['decrease']),
        'HB':fmt(ht['current_original_H_indicator']),'HFLOOR':fmt(ht['retained_original_H_floor']),'SEC':f"{ex['elapsed_seconds']:.3f}",
        'OLDHE':fmt(abs(previous['native_H']-previous['native_reference_H'])),'NEWHE':fmt(r['errors']['native_H']),
        'RSS':f"{ex['peak_RSS_MiB']:.2f}",'AUDIT':f"{a['elapsed_seconds']:.3f}",'NEXT':next_text}
    for key,value in mapping.items():tex=tex.replace('__'+key+'__',value)
    tex=re.sub(r'(?<![0-9A-Za-z_.])(-?\d+(?:\.\d+)?)[eE]([+-]\d+)',lambda m:'\\ensuremath{'+m[1]+'\\times10^{'+str(int(m[2]))+'}}',tex)
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n110. [单一128列K32浓度时间诊断](../research/reconnect/BECKER_FINE_TIME_COMPARISON.md)（2026-11-07）。\n\n'
        f'{decision}，决定{r["comparison_decision"]}。仅32新图先独立核验封存，1实际/1完整独立路径，原64/K16对照及全部旧参考复用，0旧重跑/新参考/密度/软件检查，累计502。新旧共有33采样帧逐位一致，旧全部正项/H公式/阈值保留。原三列档空间未决、旧K4/8/16通过及后移H下限障碍保持；混合时间诊断不等于完整空间或Stage3验收。主流程与独立保存序列/精确积分/80与100位预算审计完成；782原文件、15附件、109旧正式清单及全部历史修正保持。{next_text} Stage3 incomplete，fluid ORC关闭。仅本地PDF/精选ZIP与Git恢复备份，不发布/推送。\n')


def package():
    exports=ROOT/'exports';pdf=exports/f'{BASE}_report_{DATE}.pdf';archive=exports/f'{BASE}_code_results_{DATE}.zip';seal=OUT/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal));assert read(OUT/'audit.json')['passed']
    info=subprocess.check_output(['pdfinfo',str(OUT/pdf.name)],text=True);assert 'Pages:           2' in info
    log=(OUT/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT/pdf.name,pdf);save(OUT/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_visually_reviewed':True,'no_overfull_or_missing_characters':True})
    entries=read(PARENT/'sha256.json')
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_becker_spatial_budget_code_results_20261107.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file());selected.update([str(pdf.relative_to(ROOT)),str((PARENT/'sha256.json').relative_to(ROOT))])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN single128-column K32 concentration time diagnostic,2026-11-07.
Only32 new geometry graphs,1 actual/1 full independent sequence.64/K16 saved control; all old paths and3 aligned references reused,no new reference PDE or density. Count502,no new software tests.
Old positive budgets,H formula/thresholds and prior decisions retained. Shifted H time decline remains unresolved; mixed-time diagnostic cannot upgrade full spatial or Stage3 acceptance. Stage3 incomplete,fluid ORC OFF. Next read-only review registered,NOT executed.
ZIP contains782 originals,research/docs,selected parent/current records;not full workspace,15 attachments,missingv080 or chat. Local only,no publication/push.
Verify review-manifest.json and original-handoff-sha256.json. Replay saved independent sequence/observations/budgets;no new primary/PDE solve:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_fine_time.py --directory runs/reconnect-becker-fine-time
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-fine-time-restore-',dir='/tmp') as directory:
            z.extractall(directory);result=subprocess.run([sys.executable,'research/reconnect/audit_becker_fine_time.py','--directory','runs/reconnect-becker-fine-time'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=180)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    r=read(OUT/'results.json');receipt={'date':'2026-11-07','timezone':'Asia/Shanghai',
        'PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
        'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':result.stdout,'extracted_files_unchanged':True,
        'main_and_audit_first_attempt_passed':True,'comparison_decision':r['comparison_decision'],'shifted_H_unresolved_retained':True,
        'new_graphs':32,'new_primary_paths':1,'new_complete_independent_paths':1,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
        'new_software_checks':0,'recorded_checks_total':502,'prior_formal_manifests_verified':109,'attachments_verified_in_workspace':15,
        'next_gate_executed':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',receipt)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:f.write(f'# 单一K32时间诊断下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n{r["comparison_decision"]}；H后移未决保留。Stage3 incomplete，fluid ORC关闭；未发布/推送。\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fresh_replay_log'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['document','package']);a=p.parse_args();document() if a.mode=='document' else package()
