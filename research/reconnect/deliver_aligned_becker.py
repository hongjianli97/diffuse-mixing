"""Two-page local report and verified recovery package for aligned Becker references."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-aligned-becker';PARENT=ROOT/'runs/reconnect-aligned-reference'
BASE='LTN_aligned_becker';DATE='20261106'
FILES=['aligned_becker_check.py','aligned_becker_protocol.json','audit_aligned_becker.py','deliver_aligned_becker.py',
       'ALIGNED_BECKER_REFERENCE_COMPARISON.md','BECKER_POINT_SPATIAL_CONTRACT.md']


def document():
    r=read(OUT/'results.json');a=read(OUT/'audit.json');ex=read(OUT/'execution.json')
    assert ex['first_attempt_passed'] and a['passed']
    for label in ('main','audit'):shutil.copyfile(f'/tmp/ltn-aligned-becker-{label}.log',OUT/f'{label}.log')
    sources=['research/reconnect/BECKER_POINT_SPATIAL_CONTRACT.md','research/reconnect/aligned_mass_reference.py','research/rebuild/becker.py',
      'research/reconnect/bounded_periodic_voronoi.py','research/reconnect/periodic_voronoi_fv.py','research/reconnect/explicit_duration_action.py',
      'research/reconnect/voronoi_time_action.py','research/reconnect/voronoi_time16_core.py','research/reconnect/t1_uniform_core.py',
      'runs/reconnect-becker-point/graph-index.json','runs/reconnect-becker-point/results.json','runs/reconnect-aligned-becker/results.json',
      'runs/reconnect-aligned-becker/audit.json']
    sources += [f'runs/rebuild-stage3-space/inputs-cells{n}.npz' for n in (32,64,128)]
    sources += [f'runs/reconnect-becker-point/k{K}/trajectory-states.npz' for K in (4,8,16)]
    save(OUT/'next-gate-spec.json',{'already_executed':False,'type':'same_backend_moving_Becker_three_column_resolution_oracle_control',
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in sources],
      'contract':'research/reconnect/BECKER_POINT_SPATIAL_CONTRACT.md','reason':'registered32-column temporal comparison resolved under complete-child reference observation; candidate spatial evidence remains missing',
      'physical_case':{'delta':.08,'D':.02,'T':.1,'material_columns':[32,64,128],'transverse_rows_fixed':8,'K_values':[4,8,16]},
      'staged_dependency':'freeze executable positive time/spatial budget and source/geometry contract before new inputs; seal all56 new graphs before scalar/mass/old-state/reference loading; never change frozen backends',
      'reuse':'original32-column saved graphs and all3 actual/independent paths; no rerun; original32/64/128 fixedmass/c0 remain unchanged',
      'new_reference':'3 fixed aligned-reference solves on original128 material parent intervals, h=M/16384 and doubled child counts, steps8192/8192/16384; complete child sums at all32/64/128 levels; monitor and old guards; no partial-cell interpolation or corrections',
      'spatial_acceptance':'both32->64 and64->128 native/common concentration and H improvement must exceed frozen sum of original numeric, empirical reference, initial/observation/geometry bridges AND K8->16 temporal indicators; unchanged H helper, strict greater-than; empirical,not certified PDE error',
      'new_candidate_graphs':56,'new_candidate_transient_runs':6,'new_complete_independent_uniformization_sequences':6,
      'old_candidate_reruns':0,'new_reference_PDE_solves':3,'new_software_checks':0,'recorded_checks_total':502,
      'main_wall_seconds_cap':2400,'reference_part_wall_seconds_cap':300,'main_peak_RSS_MiB_cap':1024,
      'stop':'on failure/unresolved/cap preserve and stop; no extra N/K/rows/reference/model or F/C fitting, clipping,normalization,threshold change',
      'old_unresolved_results_and_current_single_grid_temporal_pass_preserved':True,'no_new_spatial_inputs_graphs_states_or_references_executed_this_turn':True,
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-06','timezone':'Asia/Shanghai','decision':r['decision'],
      'new_single_grid_temporal_comparison_resolved':r['new_temporal_comparison_resolved'],'old_two_unresolved_results_reclassified':False,
      'new_reference_PDE_solves':3,'new_candidate_transient_runs':0,'new_software_checks':0,'recorded_checks_total':502,
      'next_spatial_control_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    rows='\n'.join(f"| {v['cells']} | {v['steps']} | {v['initial_mean_bridge']:.9g} | {v['terminal_aggregation_rounding_bridge']:.9g} |" for v in r['reference_rows'])
    guards=read(OUT/'reference-runs.json');guardrows='\n'.join(f"| {v['cells']} | {v['steps']} | {v['max_relative_mass_drift']:.9g} | {v['steps_without_sufficient_positivity']}/{v['steps']} |" for v in guards)
    report=f'''# 物质对齐Becker参考：原候选时间步改善已可分辨

2026-11-06（Asia/Shanghai）。按[上一轮登记](../../runs/reconnect-aligned-reference/next-gate-spec.json)仅运行3组参考；[执行计账协议](aligned_becker_protocol.json)在新物理输入和网格前固定。δ=.08、D=.02、T=.1，原32物质列×8行和K4/8/16候选完全复用。

**新登记决定：{r['decision']}。** 在原H公式及预先固定正项预算下，状态与H时间差的下降均可分辨。旧两次未决及其原始预算不改，不追认旧通过。本结果是单候选空间网格的时间控制，不是Stage3通过，也不认证连续PDE误差界。

| 量 | 本轮保存值 |
|---|---:|
| 参考空间差质量范数 | {r['reference_space_indicator_absolute']:.12g} |
| 参考时间差质量范数 | {r['reference_time_indicator_absolute']:.12g} |
| 全部正项状态预算 | {r['budget_absolute']:.12g} |
| 原H公式给出的指标 | {r['H_indicator_unchanged_formula']:.12g} |
| 保存的H时间差下降 | {r['H_decrease_saved']:.12g} |
| H下降/指标 | {r['H_decrease_saved']/r['H_indicator_unchanged_formula']:.9g} |
| 原K16候选相对本轮参考误差 | {r['observations'][-1]['native_error_relative']:.12g} |

## 改变的是参考网格与完整积分观测

原32物质区间不动。h=精确总质量/16384，各区间子数ceil(m_i/h)，得到16395格，下一层每区间子数翻倍为32790格；三组为(16395,8192)/(32790,8192)/(32790,16384)。质量/初值均不重归一化，参考内面不提供给候选几何。原候选及旧参考源码字节不变。

原参考为不对齐均匀细格，整格均值不能唯一决定部分积分；此次父区间都是完整子区间之并集。生产观测仍按Σm_j c_j/原m_i计算，独立Fraction完整积分用于核验与舍入预算，不替换状态。初值桥采用“精确子积分与保存粗c0之差范数 + 生产汇总与精确积分之差范数”的非负和；末态汇总及质量几何缺陷也按预先固定倍率计入，没有抵消或删项。

| 参考格数 | 步数 | 初值桥（含汇总舍入） | 末态汇总舍入桥 |
|---:|---:|---:|---:|
{rows}

本轮两档所有子质量与保存端点精确差一致，父质量闭合缺陷为0。这是数据核验结果，不是通过归一化强制得到。旧均匀参考初值表示差仍保持约5.90e-9和1.09e-9；新观测的约1e-16桥来自不同、预先固定的完整积分网格，不能写成删除了旧预算。

## 守卫、时间格式与数值限制

| 参考格数 | 步数 | 最大相对质量漂移 | CN充分非负条件未满足步数 |
|---:|---:|---:|---:|
{guardrows}

三组每步原始有限性、质量漂移≤1e-9和初值极值±1e-10守卫均通过，无裁剪、重归一化或系数拟合。但CN右矩阵充分非负条件在全部步均未满足；monitor是执行前固定的策略，此处只能报告原始轨迹守卫通过，不能宣称无条件最大值原理。最高档质量漂移约8.87e-11，细化差包含浮点累积影响，不把它们都解释为空间/时间截断误差或严格误差界。

旧H数值直接沿用；当前平台K4重算H差2.22e-16单独记录，受原数值指标约束，不改H判据。K8/K16重算差为0。

## 独立审计与保全

独立扫描全部保存子区间，以精确有理数核对网格构造、嵌套、完整质量积分、质量闭合、初值和末态汇总误差；80/100位独立预算给出相同分辨结论。候选H也以精确质量方差核验。审计只针对保存端点、守卫记录和预算，未运行第4个参考或第二条独立PDE轨迹。

主流程首次通过，{ex['elapsed_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB；审计首次通过，{a['elapsed_seconds']:.3f}秒。新增3个参考解、0候选图/传播、0软件检查，累计502。782原文件、15附件、106份旧正式清单及所有历史失败恢复记录保持。

## 下一项与交付

只登记[同后端32/64/128列最小空间对照](BECKER_POINT_SPATIAL_CONTRACT.md)，本轮未执行。原32列路径复用，只补64/128两档、固定8行和K4/8/16，56图、6实际/6完整独立路径。原128父区间需另行固定3组对齐参考，避免把本轮32父区间参考的部分格均值当作新格精确积分；先封存几何，再加载质量/标量。空间改善比较必须同时计入K8→16时间差和原正项/H指标，不自动增加K或空间档。

通过空间对照也仍须同后端oracle/kernel/ledger三密度来源闭合；不把DNS、T5/T6或动态ORC新增为Stage3前置条件。Stage3 incomplete，fluid ORC关闭，κ未算。两页PDF与代码/精选结果ZIP仅本地交付，不发布/推送；不是39GB完整镜像，不含15附件、未取回v080或聊天。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/audit_aligned_becker.py \\
  --directory runs/reconnect-aligned-becker
```
该入口复核保存数据，不重新积分。全新解压及Git恢复验证记录以交付JSON和本地恢复日志为准。
'''
    with (ROOT/'research/reconnect/ALIGNED_BECKER_REFERENCE_COMPARISON.md').open('x') as f:f.write(report)
    tex=r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-06}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 物质对齐Becker参考核验}\\[4pt]{\large 原候选时间步改善已可分辨}\\[3pt]2026年11月6日（Asia/Shanghai）\end{center}
\section*{1. 本轮结果与适用范围}
按已登记方案仅运行3组参考，$\delta=.08,D=.02,T=.1$；原32列$\times$8行候选K4/8/16全部复用。
原H公式及固定正项预算下，状态与H时间差下降均可分辨，H下降约为指标的\textbf{12.34倍}。
\textbf{这是单空间网格的时间控制，Stage3仍未完成。}旧两次未决和原预算保持，不追认旧通过。
\begin{center}\begin{tabular}{lr}\toprule
量&本轮值\\\midrule
参考空间差质量范数&$5.51989040\times10^{-10}$\\
参考时间差质量范数&$6.58929270\times10^{-11}$\\
全部正项状态预算&$1.39115676\times10^{-9}$\\
原公式H指标&$4.66888702\times10^{-8}$\\
保存的H时间差下降&$5.76279600\times10^{-7}$\\
原K16相对本轮参考误差&$4.13061997\times10^{-4}$\\\bottomrule
\end{tabular}\end{center}
\section*{2. 完整物质积分解决了当前观测障碍}
原32物质区间不动，$h=M/16384$，父区间子数$\lceil m_i/h\rceil$。
两层为16395、32790格，细层各父子数翻倍；三组步数8192/8192/16384。
父观测为完整子积分
\[\bar C_i=\frac{\sum_{J_j\subset K_i}m_jc_j}{m_i}.\]
子质量、初值不归一化；原候选和旧参考代码不变。独立Fraction积分只用于核验及舍入计账，不替换状态。
初值桥保留精确子积分与原粗c0差、生产汇总舍入两项的范数和；末态及几何质量差仍作为非负项。
\begin{center}\begin{tabular}{rrr}\toprule
格数/步数&初值桥&末态汇总桥\\\midrule
16395/8192&$1.7024\times10^{-16}$&$9.8784\times10^{-17}$\\
32790/8192&$1.8917\times10^{-16}$&$1.0535\times10^{-16}$\\
32790/16384&$1.8917\times10^{-16}$&$8.1686\times10^{-17}$\\\bottomrule
\end{tabular}\end{center}
质量几何缺陷经精确核验为0，未强制归一化。旧不对齐参考约$10^{-9}$的初值表示差仍保存在旧结果中；
此次使用另行固定的网格和观测，不等于删去旧预算。经验细化差不是认证PDE误差界。
\newpage
\section*{3. 原始守卫通过，不声明CN无条件极值保持}
CN采用执行前固定的monitor策略。三组每步非有限数、相对质量漂移$\le10^{-9}$、
初值极值$\pm10^{-10}$守卫均通过，无裁剪、重归一化或F/C拟合。
\begin{center}\begin{tabular}{rrr}\toprule
格数/步数&最大相对质量漂移&充分非负条件未满足\\\midrule
16395/8192&$1.2475\times10^{-11}$&8192/8192步\\
32790/8192&$4.1359\times10^{-11}$&8192/8192步\\
32790/16384&$8.8684\times10^{-11}$&16384/16384步\\\bottomrule
\end{tabular}\end{center}
CN右矩阵的充分非负条件在全部步未满足。只能报告当前原始轨迹守卫通过，
不能宣称任意时间步保持最大值。细化差还包含浮点累积影响，不全归因于截断误差。
旧K4的H在当前平台重算差$2.22\times10^{-16}$，单独记录；比较沿用原保存值和原数值指标。
\section*{4. 独立审计与旧文件保全}
逐个子区间用精确有理数核验构造、嵌套、质量、完整积分及初末态舍入桥，
独立80/100位预算确认相同判别。候选H以精确质量方差核验。
这是保存端点及守卫记录审计，没有第4个参考或独立全时间PDE重跑。

主过程首次通过：90.439秒/62.89MiB；独立审计首次通过：17.237秒。
新增3参考解、0候选图/传播、0软件检查，累计502。
782原文件、15附件、106旧正式清单和历史失败恢复记录保持。
旧H指标$9.46809136\times10^{-7}$的未决保留，新结果单独登记。
\section*{5. 下一步只补最小空间控制}
同后端原32列路径复用，只补64/128列，固定8行和K4/8/16。
新56图先封存后读质量/标量，6实际和6完整独立路径。
参考需以原128区间为父另行固定3组，以完整子积分同时观测32/64/128列，避免重现部分格问题。
两相邻空间收益须超过数值、经验参考、初值/观测/几何正项，并计入K8到16时间差；H公式不变。
失败或未决停止，不扩N/K/参数扫描。本轮未执行此空间项。

之后仍须同后端oracle/kernel/ledger三密度来源闭合。
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未计算。}
\section*{6. 本地交付与恢复}
ZIP含代码/文档、782原文件、精选父例和本轮结果，非39GB全镜像；不含15附件、未取回v080或聊天。
未发布或推送。保存数据复核：
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_aligned_becker.py \
--directory runs/reconnect-aligned-becker
\end{verbatim}
\end{document}
"""
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n107. [物质对齐Becker参考：保存候选时间改善已分辨](../research/reconnect/ALIGNED_BECKER_REFERENCE_COMPARISON.md)（2026-11-06）。\n\n'
      '完成冻结的(16395,8192)/(32790,8192)/(32790,16384)三组参考，原32列×8行K4/8/16候选复用，0新候选/软件检查、累计502。完整子积分使初值桥约1.7–1.9e-16；原H指标4.668887e-8低于保存下降5.762796e-7，单网格时间改善可分辨，K16相对误差4.130620e-4。两次旧未决/H预算保持，不追认。三组原守恒/极值守卫通过，但CN充分非负条件在所有步未满足，monitor不作无条件最大值声明；最高质量漂移8.87e-11。独立精确积分/几何质量与80/100位预算审计首次通过，主流程90.439秒首次通过。782原文件、15附件、106旧正式清单保持。下一项仅登记同后端32/64/128列最小空间控制，固定8行，原32路径复用，只补56图/6实际及独立路径和以128父区间为基的3个参考解，空间指标必须计入时间差；本轮未执行。Stage3与三密度闭合仍未完成，fluid ORC关闭。仅本地PDF/代码ZIP和Git恢复备份，不发布/推送。\n')


def package():
    exports=ROOT/'exports';pdf=exports/f'{BASE}_report_{DATE}.pdf';archive=exports/f'{BASE}_code_results_{DATE}.zip';seal=OUT/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal))
    assert read(OUT/'audit.json')['passed'] and read(OUT/'execution.json')['first_attempt_passed']
    info=subprocess.run(['pdfinfo',str(OUT/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           2' in info
    log=(OUT/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT/pdf.name,pdf);save(OUT/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_visually_reviewed':True,'no_overfull_or_missing_characters':True})
    parent=PARENT/'sha256.json';entries=read(parent)
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    entries[str(parent.relative_to(ROOT))]=digest(parent)
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_aligned_reference_code_results_20261106.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file());selected.update([str(pdf.relative_to(ROOT)),str(parent.relative_to(ROOT))])
    # These existing frozen column inputs are also sufficient to preserve next-gate provenance.
    selected.update(f'runs/rebuild-stage3-space/inputs-cells{n}.npz' for n in (32,64,128))
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN aligned Becker reference comparison,2026-11-06.
Three predeclared independent reference solves completed; saved candidate paths untouched. New temporal comparison resolved under original H formula and fixed positive budgets; old two unresolved results preserved.
Reference grids16395/32790; steps8192/8192/16384. CN monitor raw guards passed, sufficient positivity condition failed at every step; no unconditional maximum-principle claim.
Independent exact child sums and80/100-digit budget audit passed; no fourth PDE or candidate rerun. Count502,0 new software tests. Stage3 incomplete,fluid ORC OFF.
Next minimal32/64/128 column spatial control is registered,NOT executed. See BECKER_POINT_SPATIAL_CONTRACT.md.
Includes research/docs,782 originals,selected parent cases/current results,not39GB workspace,15 attachments,missingv080 or chat.
Only local delivery,no publishing/pushing. Verify review-manifest.json and original-handoff-sha256.json.
Saved-data replay (no PDE):
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_aligned_becker.py --directory runs/reconnect-aligned-becker
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-aligned-becker-restore-',dir='/tmp') as directory:
            z.extractall(directory);replay=subprocess.run([sys.executable,'research/reconnect/audit_aligned_becker.py','--directory','runs/reconnect-aligned-becker'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    delivery={'date':'2026-11-06','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
      'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
      'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':replay.stdout,'extracted_files_unchanged':True,
      'main_and_audit_first_attempt_passed':True,'new_reference_PDE_solves':3,'new_candidate_transient_runs':0,'new_software_checks':0,
      'recorded_checks_total':502,'new_temporal_comparison_resolved':True,'old_two_unresolved_results_preserved':True,
      'next_spatial_control_executed':False,'stage3':'incomplete','fluid_orc_enabled':False,'prior_formal_manifests_verified':106,
      'attachments_verified_in_workspace':15,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',delivery)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:f.write(f'# 对齐Becker参考下载\n\n[两页PDF]({pdf.name}) · [代码与结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n单网格时间比较已分辨，Stage3未完成、fluid ORC关闭；未发布/推送。\n')
    print(json.dumps({k:v for k,v in delivery.items() if k!='fresh_replay_log'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['document','package']);args=parser.parse_args()
    document() if args.mode=='document' else package()
