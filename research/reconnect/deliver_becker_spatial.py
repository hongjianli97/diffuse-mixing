"""Local spatial-study report/plot, bounded next gate and verified recovery ZIP."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-becker-spatial';PARENT=ROOT/'runs/reconnect-aligned-becker'
BASE='LTN_becker_spatial';DATE='20261106'
FILES=['becker_spatial_core.py','becker_spatial_study.py','becker_spatial_protocol.json','BECKER_SPATIAL_BUDGET.md',
       'audit_becker_spatial.py','deliver_becker_spatial.py','BECKER_SPATIAL_COMPARISON.md','BECKER_DENSITY_SOURCE_PROPOSAL.md']


def next_gate(r):
    sources=['runs/reconnect-becker-spatial/results.json','runs/reconnect-becker-spatial/audit.json','research/reconnect/BECKER_SPATIAL_BUDGET.md',
      'docs/CODEX_TASK.md','docs/METHOD.md','research/reconnect/T3_COVERAGE_REVIEW.md']
    if r['registered_spatial_control_passed']:
        gate={'type':'one_saved_Becker_density_source_discriminator_after_two_bounded_adapter_fixtures',
          'reason':'same-backend moving Becker oracle spatial evidence completed; source isolation is the remaining registered Stage3 branch',
          'contract':'research/reconnect/BECKER_DENSITY_SOURCE_PROPOSAL.md',
          'physical_case':{'columns':128,'rows':8,'delta':.08,'D':.02,'T':.1,'K_values':[4,8,16]},
          'staged_dependency':'freeze source adapters,exact budget and two bounded software fixtures before new density arrays. Reuse geometry; seal56 new coefficient packets before c0/reference evaluation. No legacy module execution or F/C fitting',
          'kernel':'1D cubic B-spline mass sum, h_i=2 times point-midpoint dual width; single reflections at given external boundaries, strict support check. Analytic continuous kernel normalization only; no discrete normalization',
          'ledger':'fixed-t0 anchor rho0_point*V0_proxy/Vt_proxy; fixed point adjacency, no F/C, reanchoring, fallback or repair. Proxy anchor mass need not equal original material mass; disclose it,never replace original m',
          'reuse':'same immutable128-column geometry,m/c0,D,boundaries,all3 aligned references and3 oracle actual/independent paths; no new oracle/reference solve',
          'source_inputs':'point positions/IDs and external boundaries plus original column masses for kernel; ledger additionally given initial point density only. No c0,terminal reference,true internal material faces or t>0 oracle density',
          'fixed_comparison':'three-step temporal controls plus original positive numerical/reference/initial/observation budgets; K16 density-source state/H differences include both modes K8->16 temporal indicators. Freeze details before densities; same original H and strict greater-than',
          'new_candidate_geometry_builds':0,'new_candidate_coefficient_packets':56,'new_candidate_graphs':56,'new_candidate_transient_runs':6,
          'new_complete_independent_uniformization_sequences':6,'new_reference_PDE_solves':0,'old_oracle_reruns':0,
          'new_software_checks':2,'recorded_checks_if_both_fixtures_pass':504,'fixture_cap':'at most16 synthetic columns and8 time samples,120s; constant kernel/affine ledger/rejection and fixed face-coefficient interface checks',
          'main_wall_seconds_cap':600,
          'stop':'failure/unresolved retained; no source-by-space matrix or additional N/K/h/anchor/reference, clipping,renormalization,F/C fit or threshold retune; this single-case discriminator is not three-source spatial admission'}
        sources += ['research/reconnect/BECKER_DENSITY_SOURCE_PROPOSAL.md','research/reconnect/explicit_duration_action.py',
          'research/reconnect/periodic_voronoi_fv.py','runs/reconnect-becker-spatial/n128/pure-inputs.json',
          'runs/reconnect-becker-spatial/n128/graph-index.json','runs/reconnect-becker-spatial/n128/material.npz']
        text='下一项仅同一128列算例的密度来源判别：先完成两个小型软件检查，再新增kernel和固定锚点无梯度ledger两来源的K4/8/16路径，oracle路径、几何、m/c0/D及参考全部复用。局部核支撑、外边界镜像和代理体积比按书面契约预先固定；不使用旧v4拟合/回退，不将代理体积密度混同真实物质均值。只补56个系数包、6实际/6独立路径，不重构几何或解PDE；单档比较仍不等于三来源空间接纳。'
    else:
        gate={'type':'read_only_saved_spatial_budget_obstruction_review','reason':'registered spatial/time control not fully resolved; retain original result and stop new simulations',
          'scope':'decompose saved failing native/common/state/H components; identify temporal,reference,numeric or spatial-sign obstruction without changing formula or original result; only propose next discriminating gate if evidence supports it',
          'new_scientific_density_arrays':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_PDE_solves':0,'new_software_checks':0,
          'stop':'no newN/K/rows/parameters/density or reference until a separate supported fixed plan; no threshold changes or retroactive pass'}
        text='下一项仅只读分解保存的未决空间/时间预算，定位时间、参考、数值或空间改善符号障碍；不新增N/K/行数/参考/密度，不修改本轮判据或追认通过。'
    gate.update({'already_executed':False,'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in sources],
      'old_results_and_current_spatial_decision_preserved':True,'recorded_checks_total':502,'main_peak_RSS_MiB_cap':1024,
      'stage3_physical_acceptance':'incomplete','all_three_density_sources_closed':False,'fluid_orc_enabled':False,'kappa_computed':False})
    gate.setdefault('main_wall_seconds_cap',120)
    save(OUT/'next-gate-spec.json',gate);return text


def document():
    r=read(OUT/'results.json');a=read(OUT/'audit.json');ex=read(OUT/'execution.json');assert a['passed'] and ex['first_attempt_passed']
    for label in ('main','audit'):shutil.copyfile(f'/tmp/ltn-becker-spatial-{label}.log',OUT/f'{label}.log')
    next_text=next_gate(r);passed=r['registered_spatial_control_passed'];decision='登记的三列档空间控制通过' if passed else '64→128列浓度改善尚未分辨，空间控制未通过'
    save(OUT/'stage3-status.json',{'date':'2026-11-06','timezone':'Asia/Shanghai','comparison_decision':r['comparison_decision'],
      'registered_moving_Becker_spatial_control_passed':passed,'new_graphs':56,'new_candidate_paths':6,'new_independent_paths':6,'new_reference_PDE_solves':3,
      'old32_reruns':0,'new_software_checks':0,'recorded_checks_total':502,'old_two_unresolved_records_reclassified':False,
      'next_gate_executed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    rows=[]
    for N in (32,64,128):
        o=r['observations'][str(N)];t=r['time_controls'][str(N)]
        rows.append(f"| {N} | {o['native_error_common_denominator']:.10g} | {o['common_error']:.10g} | {abs(o['native_H']-o['native_reference_H']):.10g} | {t['time_control_resolved']} |")
    spatial=[]
    for row in r['adjacent_spatial_comparisons']:
        for key in ('native','common','native_H','common_H'):
            spatial.append(f"| {row['coarse_columns']}→{row['fine_columns']} | {key} | {row['gains'][key]:.10g} | {row['fixed_indicators'][key]:.10g} | {row['decisions'][key]} |")
    ref=read(OUT/'reference/mesh-contract.json');guards=read(OUT/'reference/reference-runs.json')
    report=f'''# 移动Becker：同一普通FV点后端的三列档空间对照

2026-11-06（Asia/Shanghai）。**{decision}**；机器决定：{r['comparison_decision']}。执行[上一轮唯一登记](../../runs/reconnect-aligned-becker/next-gate-spec.json)，[正项公式](BECKER_SPATIAL_BUDGET.md)和实现均在新输入/图/参考前冻结。δ=.08、D=.02、T=.1，横向固定8行，32/64/128物质列和K4/8/16。原32列实际及独立路径仅复制读取，不重跑。

| 列数 | 原生浓度误差/共同σ32 | 共同32列误差/σ32 | 原生H参考误差 | 三档时间控制分辨 |
|---:|---:|---:|---:|---|
{chr(10).join(rows)}

浓度分母固定为原32列初始质量波动范数，报告另含各档自身分母误差；参考为真实物质区间平均，不是点值或点Voronoi单元平均。此为一维列细化，固定8行不声称二维联合细化或一般空间阶。

## 两相邻空间档的全部固定判别

| 相邻列档 | 指标 | 实测改善 | 固定正项指标 | 决定 |
|---|---|---:|---:|---|
{chr(10).join(spatial)}

本轮三档时间控制均可分辨，32→64列四项及64→128列两项H比较均通过。64→128列原生浓度改善8.32439e-5，小于固定指标1.10750e-4；共同观测改善7.85970e-5，也小于1.06091e-4。误差下降的方向与“改善已超过不确定度”是不同结论；当前停止扩大算例。主流程与独立审计完成不等于物理空间门通过。

原生/共同浓度和H均用严格大于规则。每档空间指标显式加入K8→16状态或H变化，再加原机器/尾项/完整独立路径、经验参考、初值/观测/几何桥；时间差呈下降不构成忽略有限时间误差的理由。共同观测按原质量完整汇总相邻列及8行，容量因子仅用于误差界，不归一化行或状态。各有符号量只用于原误差差，正项预算不相消。

## 几何、传播与参考的来源隔离

新56个图逐个独立重建边界/平分线顶点、共享面、全单元半平面及非最近周期镜像排除证明。全部封存后才加载质量/c0/旧状态/参考。内部物质参考面不提供给候选构造器，点仍为原质量标签中点轨迹，x外端点移动、y单位周期、外面零标量通量。所有原普通阻力、几何、指数作用和完整统一化源码不改。

新64/128共6条实际与6条完整独立路径，原原始守恒/极值/方差/符号/尾项检查保持；旧32路径和保存H沿用。独立审计核对封存前几何证明，重新从每条共享面计算阻力/精确率、最小尾截断、原始Poisson权重和完整序列，并检查共享通量/对称算子作用。没有删正面、裁剪、概率行归一化、F/C拟合或状态外推。

参考以原128物质区间为父，得到{ref['base_cells']}/{ref['fine_cells']}格，步数8192/8192/16384，仅3解。一个细解以完整子积分同时观测32/64/128层，避免部分细格问题。初值、汇总、子质量几何差继续作正项；精确Fraction积分用于核验，不修补状态。三组CN monitor原始守卫通过，充分非负条件未满足的步数为{[v['steps_without_sufficient_positivity'] for v in guards]}，不能声称无条件最大值原理。最大参考相对质量漂移{max(v['max_relative_mass_drift'] for v in guards):.6g}；经验细化差包含舍入，不是认证连续误差界。

## 审计、保全与剩余项

独立保存端点完整质量积分、参考各层观测、候选误差范数及80/100位正项预算复核通过。主过程首次通过：{ex['elapsed_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB；独立审计首次通过：{a['elapsed_seconds']:.3f}秒。仅56新图、6实际/6独立路径、3参考解；0新软件检查、累计502。782原文件、15附件、107份旧正式清单及所有旧失败恢复记录保持；旧两个未决和上轮32列时间通过都保留，未追认或覆盖。

{next_text}

Stage3仍incomplete，fluid ORC关闭，κ未算。当前仍不能宣称同后端oracle/kernel/ledger三密度来源闭合或生产默认时间接口已完成迁移。DNS、T5/T6和动态ORC不新增为本阶段前提。

## 本地交付

两页PDF、收敛图、研究代码/文档、782原文件及精选父例/本轮结果ZIP；不是39GB完整镜像，不含15附件、未取回v080或聊天。仅本地提交和恢复备份，不发布或推送。CRC/SHA、全新解压保存数据重放见交付JSON。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/audit_becker_spatial.py \\
  --directory runs/reconnect-becker-spatial
```
保存数据审计重建独立序列，但不重新运行候选指数作用、构图或参考PDE。
'''
    with (ROOT/'research/reconnect/BECKER_SPATIAL_COMPARISON.md').open('x') as f:f.write(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ns=[32,64,128];fig,axes=plt.subplots(1,2,figsize=(8.2,2.65),layout='constrained')
    for key,label in [('native_error_common_denominator','Native'),('common_error','Common32')]:axes[0].loglog(ns,[r['observations'][str(n)][key] for n in ns],'o-',label=label)
    for pre,label in [('native','Native'),('common','Common32')]:axes[1].loglog(ns,[abs(r['observations'][str(n)][pre+'_H']-r['observations'][str(n)][pre+'_reference_H']) for n in ns],'o-',label=label)
    for ax in axes:ax.set_xticks(ns,labels=list(map(str,ns)));ax.set_xlabel('Material columns (8 rows fixed)');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
    for ax in axes:ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel('Mass L2 error / original sigma32');axes[1].set_ylabel('Absolute H reference error')
    fig.savefig(OUT/'spatial-comparison.png',dpi=170);plt.close(fig)
    table='\n'.join(f"{N}&{r['observations'][str(N)]['native_error_common_denominator']:.6g}&{r['observations'][str(N)]['common_error']:.6g}\\\\" for N in ns)
    verdicts='\n'.join(f"{row['coarse_columns']}--{row['fine_columns']}&{key.replace('_',' ')}&{row['gains'][key]:.5g}&{row['fixed_indicators'][key]:.5g}\\\\" for row in r['adjacent_spatial_comparisons'] for key in ('native','common','native_H','common_H'))
    tex=r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,graphicx,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-06}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 移动Becker三列档空间对照}\\[4pt]{\large __DECISION__}\\[3pt]2026年11月6日（Asia/Shanghai）\end{center}
\section*{1. 同一后端、同一算例}
$\delta=.08,D=.02,T=.1$，固定8行和K4/8/16。
原32列全部路径复用，只补64/128列。质量平均状态、原质量/c0、普通阻力、几何和传播源码保持。
这是一维列细化，不声明二维联合细化或一般空间阶。
\begin{center}\begin{tabular}{rrr}\toprule
列数&原生浓度误差/共同$\sigma_{32}$&共同32列误差/$\sigma_{32}$\\\midrule
__TABLE__
\bottomrule\end{tabular}\end{center}
\includegraphics[width=\textwidth]{spatial-comparison.png}
\section*{2. 两相邻档的固定正项比较}
\begin{center}\begin{tabular}{llrr}\toprule
列档&量&实测改善&固定指标\\\midrule
__VERDICTS__
\bottomrule\end{tabular}\end{center}
改善须严格大于指标。空间预算显式加入K8到16的状态或H变化，保留数值、经验参考、
初值/观测/几何各正项，不因时间差下降而忽略时间误差。共同观测只汇总完整原质量积分。
32到64列四项与64到128列两项H比较通过；后者两项浓度改善仍未超过指标。
\newpage
\section*{3. 几何先封存；参考完整物质对齐}
新56图全部在读取质量/c0/旧状态/参考前独立核验并封存。
候选只接收点、给定密度/D及移动外边界，不接收内部物质参考面。
所有共享面、全单元半平面及非最近周期镜像排除证明留档，外面零通量、y单位周期。

以原128区间为父，参考两层__BASE__/__FINE__格，步数组合8192/8192/16384，仅3解。
完整子积分同时观察32/64/128列，精确有理数核对初值、终态、质量闭合及浮点桥。
CN使用预先固定monitor，原始守恒/极值守卫通过；充分非负条件未满足步数__CN__。
不能宣称无条件最大值原理，经验细化差不是认证PDE误差界。
\section*{4. 独立审计与数据保全}
审计核对封存前全单元几何证明，并独立重建每面阻力、精确交换率、最小尾截断、
原始权重和完整独立传播序列。共享通量/对称作用、原始质量/极值/方差、
参考完整积分及80/100位误差预算通过。不重新构图、运行候选指数作用或解参考PDE。

主流程首次通过：__SECONDS__秒/__RSS__MiB；审计首次通过：__AUDIT__秒。
56新图、6实际/6完整独立路径、3参考；0新软件检查，累计502。
782原文件、15附件、107旧正式清单与所有历史失败恢复保持。
旧两次未决和上轮32列时间通过不改，不追认旧结果。
\section*{5. 下一项的边界}
__NEXT__

\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。}
三密度来源尚未闭合，不把DNS、T5/T6或动态ORC新增为本阶段前提。
\section*{6. 本地报告和恢复包}
ZIP含研究代码/文档、782原文件、精选父例和本轮结果，不是39GB全镜像；
不含15附件、未取回v080或完整聊天。未发布或推送。保存数据复核：
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_becker_spatial.py \
--directory runs/reconnect-becker-spatial
\end{verbatim}
\end{document}
"""
    values={'DECISION':decision,'TABLE':table,'VERDICTS':verdicts,'BASE':ref['base_cells'],'FINE':ref['fine_cells'],
      'CN':' / '.join(str(v['steps_without_sufficient_positivity']) for v in guards),'SECONDS':f"{ex['elapsed_seconds']:.3f}",
      'RSS':f"{ex['peak_RSS_MiB']:.2f}",'AUDIT':f"{a['elapsed_seconds']:.3f}",'NEXT':next_text}
    for key,value in values.items():tex=tex.replace('__'+key+'__',str(value))
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write(f'\n\n108. [移动Becker同后端三列档空间对照](../research/reconnect/BECKER_SPATIAL_COMPARISON.md)（2026-11-06）。\n\n'
      f'{decision}，决定{r["comparison_decision"]}。原32列路径复用，只补64/128列、固定8行及K4/8/16；56新图先封存，6实际/6独立路径，原128父区间3个对齐参考。两相邻档原生/共同浓度和H改善按固定正项比较，显式计入K8→16时间差；不改原H/阈值、质量/状态或旧代码。主计算和独立几何证明/共享面率/完整序列/精确积分/80与100位预算审计首次通过；0新软件检查、累计502。782原文件、15附件、107旧正式清单保持。{next_text} Stage3 incomplete，fluid ORC关闭。仅本地PDF/代码ZIP与Git恢复备份，不发布/推送。\n')


def package():
    exports=ROOT/'exports';pdf=exports/f'{BASE}_report_{DATE}.pdf';archive=exports/f'{BASE}_code_results_{DATE}.zip';seal=OUT/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal));assert read(OUT/'audit.json')['passed']
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
    with zipfile.ZipFile(exports/'LTN_aligned_becker_code_results_20261106.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file());selected.update([str(pdf.relative_to(ROOT)),str(parent.relative_to(ROOT))])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN same-backend moving Becker column-space control,2026-11-06.
Only64/128 columns added;32-column paths reused.8rows,K4/8/16 fixed;56 new graphs,6 actual/6 independent paths,3 aligned references on original128 parents.
Positive spatial budgets include observed8->16 time differences,original numerical/reference/initial/observation/geometry terms. See results.json for pass/failure scope; no general order or certified continuum bound.
Independent pre-scalar all-cell geometry receipts,face/rate/full-sequence reconstruction,exact complete-child reference observations and80/100-digit budget audited.
Old unresolved and prior32 temporal pass preserved. Count502,0 new software tests. Stage3 incomplete,fluid ORC OFF. Next gate registered,NOT executed.
Contains research/docs,782 originals,selected parent cases/current results;not39GB workspace,15 attachments,missingv080 or chat. Local only,no publishing/pushing.
Verify review-manifest.json and original-handoff-sha256.json. Saved-data replay,not new reference or primary candidate propagation:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_spatial.py --directory runs/reconnect-becker-spatial
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-becker-spatial-restore-',dir='/tmp') as directory:
            z.extractall(directory);replay=subprocess.run([sys.executable,'research/reconnect/audit_becker_spatial.py','--directory','runs/reconnect-becker-spatial'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=180)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    r=read(OUT/'results.json');delivery={'date':'2026-11-06','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
      'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
      'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':replay.stdout,'extracted_files_unchanged':True,
      'main_and_audit_first_attempt_passed':True,'new_graphs':56,'new_candidate_transient_runs':6,'new_complete_independent_paths':6,'new_reference_PDE_solves':3,
      'old32_reruns':0,'new_software_checks':0,'recorded_checks_total':502,'comparison_decision':r['comparison_decision'],
      'old_decisions_preserved':True,'next_gate_executed':False,'stage3':'incomplete','fluid_orc_enabled':False,'prior_formal_manifests_verified':107,
      'attachments_verified_in_workspace':15,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',delivery)
    with (exports/f'{BASE}_downloads_{DATE}.md').open('x') as f:f.write(f'# Becker空间对照下载\n\n[两页PDF]({pdf.name}) · [代码与结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n{r["comparison_decision"]}。Stage3未完成、fluid ORC关闭；未发布/推送。\n')
    print(json.dumps({k:v for k,v in delivery.items() if k!='fresh_replay_log'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['document','package']);a=p.parse_args();document() if a.mode=='document' else package()
