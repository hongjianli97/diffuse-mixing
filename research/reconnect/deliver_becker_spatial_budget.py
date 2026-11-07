"""Local report and recovery package for the read-only spatial-budget review."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-becker-spatial-budget'
PARENT=ROOT/'runs/reconnect-becker-spatial'
BASE='LTN_becker_spatial_budget';DATE='20261107'
FILES=['becker_spatial_budget_review.py','becker_spatial_budget_review_protocol.json','audit_becker_spatial_budget.py',
       'becker_spatial_budget_completion.py','audit_becker_spatial_budget_completion.py','BECKER_FINE_TIME_PROBE_CONTRACT.md',
       'BECKER_SPATIAL_BUDGET_REVIEW.md','deliver_becker_spatial_budget.py']


def document():
    r=read(OUT/'results.json');c=read(OUT/'planning-correction.json');a=read(OUT/'completion-audit.json');assert a['passed']
    for name in ('main','audit','completion','completion-audit'):
        shutil.copyfile(f'/tmp/ltn-becker-spatial-budget-{name}.log',OUT/f'{name}.log')
    sources=['research/reconnect/BECKER_FINE_TIME_PROBE_CONTRACT.md','runs/reconnect-becker-spatial-budget/planning-correction.json',
             'runs/reconnect-becker-spatial-budget/completion-audit.json','runs/reconnect-becker-spatial/results.json',
             'research/reconnect/BECKER_SPATIAL_BUDGET.md','research/reconnect/becker_point_core.py',
             'research/reconnect/bounded_periodic_voronoi.py','research/reconnect/audit_bounded_geometry.py',
             'research/reconnect/explicit_duration_action.py']
    save(OUT/'next-gate-spec.json',{'type':'single_finest_Becker_K32_concentration_time_diagnostic','already_executed':False,
        'contract':sources[0],'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in sources],
        'physical_case':{'columns':128,'rows':8,'delta':.08,'D':.02,'T':.1,'new_K':32},
        'reuse':'all saved32/64/128 K4/8/16 paths,m,c0,three128-parent aligned references;64/K16 remains unchanged comparison control',
        'staging':'freeze implementation and exact positive accounting before new inputs; independently verify and seal32 midpoint graphs before loading m,c0,oldstates or reference;no internal material faces',
        'new_geometry_rule':'same label-midpoint trajectories and rational sample times; K32 wrapper only,no mutation of old COUNTS',
        'budget':'B128*=oldB128+2e32+new transverse bridge; J128*=oldJ128+new common rounding+new capacity bound. Keep64 S and all old positive terms; replace only128 observed8->16 difference by measured16->32. Original sigma/H/strict comparison unchanged',
        'concentration_diagnostic_pass':'native and common actual mixed-time gains exceed respective fixed positive indicators; state8->16 minus16->32 > B128*; raw guards and complete independent sequence pass',
        'H_policy':'report native/common H spatial comparison and shifted H decrease; algebraic retained-floor obstruction remains, no threshold reduction or claim of complete H temporal pass',
        'new_candidate_graphs':32,'new_candidate_transient_runs':1,'new_complete_independent_uniformization_sequences':1,
        'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_density_sources':0,'new_software_checks':0,'recorded_checks_total':502,
        'main_wall_seconds_cap':2400,'main_peak_RSS_MiB_cap':1024,
        'stop':'preserve failure/unresolved/cap;no automatic64/K32,K64,space/reference refinement or density branch; no clipping,normalization,extrapolated state,F/C fitting,H or threshold changes',
        'new_full_spatial_or_Stage3_pass_allowed':False,'old_results_preserved':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-07','timezone':'Asia/Shanghai','decision':c['decision'],
        'original_spatial_decision':r['parent_decision_unchanged'],'original_spatial_case_passed':False,
        'planning_inference_corrected_append_only':True,'first_arithmetic_audit_did_not_prove_planning_necessity':True,
        'new_graphs':0,'new_candidate_paths':0,'new_reference_PDE_solves':0,'new_software_checks':0,'recorded_checks_total':502,
        'next_gate_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    rows=[v for v in r['comparisons'] if v['decision']=='comparison_inconclusive']
    table='\n'.join(f"| {v['observable']} | {v['gain']:.9g} | {v['original_budget']:.9g} | {v['temporal']['float']:.9g} | {100*v['temporal_fraction_of_original_budget']:.6f}% | {v['all_other_positive_terms']['float']:.9g} |" for v in rows)
    plans=[]
    for row in c['corrected_planning']:
        for choice in row['choices']:
            plans.append(f"| {row['observable']} | {choice['new_K32_columns']} | {choice['sufficient_increment_sum_ceiling_if_new_numeric_bridge_zero']:.9g} | {choice['observed_old_increment_over_ceiling']:.6f} |")
    report=f'''# Becker空间未决预算：只读复核与最小时间诊断方案

2026-11-07（Asia/Shanghai）。**当前未决由保存的时间变化指标主导，原空间控制仍未通过。** 本轮0新图、0新传播、0新PDE、0新密度，软件检查累计502不变；没有重新运行旧502项测试。

## 实测分解

| 64→128列量 | 保存改善 | 原固定指标 | K8→16时间项 | 时间占比 | 其余项总和 |
|---|---:|---:|---:|---:|---:|
{table}

即使仅保留时间项，指标仍高于改善；参考加密或降低求解器舍入并不能单独消除当前障碍。这是预算归因，不是把连续PDE误差分解为物理原因。所有旧公式、正项及有符号浮点累加差保留；累加差只解释逐项相加与保存浮点数的末位差，不能削减物理预算。H的非线性多项式不拆成相加的因果误差。

原生其他项中，机器指标约1.27675e-8，参考空间约8.06838e-9，参考时间约4.29358e-10；其他桥和独立路径差更小。八项空间判别全部重现，仍是六项改善已分辨、两项浓度未决。三档保存状态差比4.00879/4.03451/3.95697仅报告为已有数据，不拟合一般阶数或外推状态。

## H判据不能机械后移

| 列数 | 保存的 \\|H8−H16\\|（任何后续下降的上限） | 原H指标下限 |
|---:|---:|---:|
|64|1.875603106e-8|6.527387312e-8|
|128|1.582723197e-8|1.751753825e-7|

对任何H32，\\|H8−H16\\|−\\|H16−H32\\|≤\\|H8−H16\\|。若保留旧正项且只增加新路径数值项，原h_indicator随预算单调不减。因此机械平移到K8/16/32的严格H下降判据必然仍未分辨；不降低其阈值或删预算。原K4/8/16时间通过是另一项保存结论，保持不变。该逻辑只说明此判据与当前分辨尺度的限制，不能证明算法物理失败或H误差不存在。

## 规划推断修正：一条路径也有判别价值

第一次复核正确重算了预算与“两档都更新”的反三角充分界，但错误把它应用到“一档不更新”，从而推断必须补64和128两条路径。第一次独立审计只核对算术，未证明该必要性推断。原代码、results.json、audit.json全部保留；planning-correction.json和completion-audit.json追加纠正，并保存精确反例：旧粗/细误差10、2，旧时间指标各5；只将细误差变为3，终态变化1，则新改善7仍大于新指标5+1=6。原错误推断不再用于决策。

正确界限：活动档集合A的终态变化之和为d_A/σ，未活动档原时间指标为d_I/σ，原其他项C保留，新正项C_add≥0。反三角不等式给G_new≥G_old−d_A/σ，因此

    2 d_A/σ + d_I/σ + C + C_add < G_old

足以保证新浓度改善超过新指标。未更新终态不改变G，其旧时间指标只出现一次。这个充分界不是必要条件、预测或新验收阈值；实际新实验仍比较实测改善与原规则指标。

| 量 | 假定更新到K32的列档 | 暂设新正项为零时d_A/σ的充分上限 | 旧活动档时间项/上限 |
|---|---|---:|---:|
{chr(10).join(plans)}

只补128列已具有正的充分空间，选为一个最细网格时间探针；不声称这是所有可能方案中成本最低者，也不承诺一次补算就通过。下一项只加128/K32一条实际与一条独立路径、32个新图，复用64/K16及全部参考。见[固定契约](BECKER_FINE_TIME_PROBE_CONTRACT.md)。该混合时间分辨率比较不能升级成共同K32三网格空间通过，H未决也不豁免。

## 复核与交付

预算主复核、80/100位独立算术核验与追加修正核验完成；精确有理数跟踪所有保存项，独立重算状态差范数。782原始文件、15附件、108份旧正式清单保持。规划逻辑修正明确留档，不称全流程无误。无新软件测试，累计502不变。

本地PDF及代码/精选结果ZIP；ZIP不是全工作区镜像，不含15上传附件、未取回v080或完整聊天。未发布/推送。保存数据复核入口：

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/audit_becker_spatial_budget_completion.py \\
  --directory runs/reconnect-becker-spatial-budget
```

Stage3 incomplete，fluid ORC关闭，κ未算。下一项尚未执行，失败或未决停止，不自动增加K64、另一条K32、空间/参考/密度扫描。
'''
    with (ROOT/'research/reconnect/BECKER_SPATIAL_BUDGET_REVIEW.md').open('x') as f:f.write(report)
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=22mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries Becker空间预算只读复核}\\[5pt]{\large 时间项占99.98\%，原空间控制仍未通过}\end{center}
\section*{1. 卡点已定位}
同一保存的32/64/128列、8行、K4/8/16、$\delta=.08,D=.02,T=.1$。
本轮没有新图、传播、参考PDE或密度数组。以下浓度量沿用原32列初始波动范数作分母。
\begin{center}\begin{tabular}{lrr}\toprule
64到128列&原生观测&共同32列观测\\\midrule
保存改善&$8.32439\times10^{-5}$&$7.85970\times10^{-5}$\\
原固定指标&$1.10750\times10^{-4}$&$1.06091\times10^{-4}$\\
K8到16时间项&$1.10729\times10^{-4}$&$1.06070\times10^{-4}$\\
时间占比&99.9808\%&99.9800\%\\
其余项总和&$2.12653\times10^{-8}$&$2.12654\times10^{-8}$\\\bottomrule
\end{tabular}\end{center}
即使仅保留时间项，指标仍超过改善。单独加密参考或降低浮点误差不能消除当前障碍。
这是预算组成分析，不能把它解释为连续PDE误差的因果分解。原六项已分辨、两项浓度未决均保持。
\section*{2. 为什么不能承诺再加一档就通过}
保存状态差比约4.009、4.035、3.957，仅作为已有观测，不拟合阶数或外推状态。
更细时间步的H下降存在原判据的分辨障碍：
\begin{center}\begin{tabular}{rrr}\toprule
列数&$|H_8-H_{16}|$&原H指标\\\midrule
64&$1.87560\times10^{-8}$&$6.52739\times10^{-8}$\\
128&$1.58272\times10^{-8}$&$1.75175\times10^{-7}$\\\bottomrule
\end{tabular}\end{center}
对任意$H_{32}$，$|H_8-H_{16}|-|H_{16}-H_{32}|\leq|H_8-H_{16}|$。
保留旧正项再加新数值项，H指标不会降低。机械平移后的严格下降条件因此无法通过；
不降低阈值、删正项或用H接近零豁免。旧K4/8/16时间通过保持，不能混同两个判别。
\section*{3. 一处规划推断已纠正并留档}
原预算分解和两档同时更新的充分界正确；将其套到单档更新，错误推断必须补两条路径。
第一次审计核对算术，未证明这一必要性推断。原代码、结果和审计保留，追加修正与核验。

精确反例：旧粗/细误差10、2，旧时间指标各5。只把细误差变成3，终态变化1；
新改善7大于新指标$5+1=6$。未更新终态没有变化，其旧指标不能再计入终态变化界。
\newpage
\section*{4. 正确界限与最小下一项}
令活动档终态变化为$d_A/\sigma$，未活动档旧时间指标为$d_I/\sigma$，
保留其他项为$C$、新增正项$C_+\ge0$。反三角不等式给出
\[
G_{\rm new}\ge G_{\rm old}-d_A/\sigma,\qquad
2d_A/\sigma+d_I/\sigma+C+C_+<G_{\rm old}.
\]
后一式足以保证新浓度比较分辨；它不是必要条件、预测或新的验收阈值。
只补128列时，暂设新正项为零的两项$d_A/\sigma$上限约
$1.50827\times10^{-5}$、$1.34010\times10^{-5}$，均为正。
实际新改善必须由新终态重算，不能沿用旧改善假装通过。

\textbf{下一项仅128列K32：32个新中点图，1条实际与1条完整独立路径。}
64列保留K16对照，全部原质量、初场、轨迹规则和三个独立参考复用；不解新参考PDE。
图先独立核验封存，再读标量和参考。实现及公式在新数据前冻结，沿用普通阻力和原传播后端。

新预算保留旧B全部项，再加新路径机器/尾项/独立差及观测桥。
只将128列的时间变化换成实测K16到32变化，64列旧时间项保持。
共同观测仍用完整原质量积分，原H公式和严格大于规则不变。
两项浓度比较、状态时间变化及原守恒/极值/方差/独立路径检查均通过，才报告浓度诊断分辨。

这是$(K_{64},K_{128})=(16,32)$的混合时间诊断，不是共同K32的三网格空间验收。
H后移未决仍保留，无论结果如何都不能直接宣布Stage3通过或开启三密度来源分支。
主流程上限2400秒、1024MiB；失败、未决或超限停止，不自动再补K32/K64、空间或参考。
\section*{5. 复核、保全与本地交付}
精确有理数预算、独立状态差范数及80/100位算术核对通过；规划修正另以反例和活动/非活动界核验。
0新软件测试，累计502；未重跑旧502项。782原文件、15附件、108旧正式清单保持。
不把规划修正隐去或称全流程无误。

PDF与代码/精选结果ZIP仅本地保存。ZIP不是全工作区镜像，不含15上传附件、未取回v080或聊天。
未发布或推送。保存数据复核入口：
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_becker_spatial_budget_completion.py \
--directory runs/reconnect-becker-spatial-budget
\end{verbatim}
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。下一项尚未执行。}
\end{document}
'''
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n109. [Becker空间未决预算只读复核](../research/reconnect/BECKER_SPATIAL_BUDGET_REVIEW.md)（2026-11-07）。\n\n'
        '时间项占未决原生/共同浓度预算99.9808%/99.9800%，原六项分辨、两项未决保持；0新图/状态/PDE/密度/软件检查，累计502。机械后移H判据受旧预算下限阻碍，不能承诺K32使完整Stage3通过。两档更新的充分界曾错误用于单档方案，原代码/结果/算术审计保留，以精确反例和追加审计纠正。下一项登记仅128/K32一条实际/独立路径和32图，64/K16控制及参考复用；固定正项、H和严格比较不改，混合时间诊断不等于三档空间验收，尚未执行。782原文件、15附件、108旧正式清单保持。Stage3 incomplete，fluid ORC关闭；仅本地PDF/精选ZIP和Git恢复备份，不发布/推送。\n')


def package():
    exports=ROOT/'exports';pdf=exports/f'{BASE}_report_{DATE}.pdf';archive=exports/f'{BASE}_code_results_{DATE}.zip';seal=OUT/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal));assert read(OUT/'completion-audit.json')['passed']
    info=subprocess.check_output(['pdfinfo',str(OUT/pdf.name)],text=True);assert 'Pages:           2' in info
    log=(OUT/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT/pdf.name,pdf);save(OUT/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_visually_reviewed':True,'no_overfull_or_missing_characters':True})
    entries=read(PARENT/'sha256.json')
    for name,h in entries.items():assert digest(ROOT/name)==h,name
    entries[str((PARENT/'sha256.json').relative_to(ROOT))]=digest(PARENT/'sha256.json')
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_becker_spatial_code_results_20261106.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file());selected.update([str(pdf.relative_to(ROOT)),str((PARENT/'sha256.json').relative_to(ROOT))])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN saved Becker spatial-budget review,2026-11-07. No new graphs,states,PDE or tests,count502. Stage3 incomplete,fluid ORC OFF.
Temporal indicators dominate the unresolved concentration budget. Original spatial decision unchanged. A planning necessity inference was wrong: initial code/results/audit retained,planning-correction.json and completion-audit.json supersede that inference. Corrected next gate is a single128/K32 diagnostic,NOT executed;no automatic full-spatial acceptance.
ZIP includes782 originals,research/docs,selected parent/current records;not full workspace,15 attachments,missingv080 or chat. Local only,no publication/push.
Verify review-manifest.json and original-handoff-sha256.json. Replay saved arithmetic only:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_spatial_budget_completion.py --directory runs/reconnect-becker-spatial-budget
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-spatial-budget-restore-',dir='/tmp') as directory:
            z.extractall(directory);result=subprocess.run([sys.executable,'research/reconnect/audit_becker_spatial_budget_completion.py','--directory','runs/reconnect-becker-spatial-budget'],cwd=directory,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    receipt={'date':'2026-11-07','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
        'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
        'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':result.stdout,'extracted_files_unchanged':True,
        'planning_inference_corrected_and_original_preserved':True,'all_review_logic_first_attempt_correct':False,
        'new_graphs_or_states_or_PDE_or_tests':0,'recorded_checks_total':502,'prior_formal_manifests_verified':108,'attachments_verified_in_workspace':15,
        'next_gate_executed':False,'stage3':'incomplete','fluid_orc_enabled':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',receipt)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:f.write(f'# 空间预算复核下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n原空间未决保留；规划推断已追加纠正。Stage3 incomplete，fluid ORC关闭；未发布/推送。\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fresh_replay_log'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['document','package']);a=p.parse_args();document() if a.mode=='document' else package()
