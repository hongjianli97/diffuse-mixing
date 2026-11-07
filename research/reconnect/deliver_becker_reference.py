"""Document and package the stopped reference check; preserve all prior evidence."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-becker-reference'
BASE='LTN_becker_reference'
DATE='20261106'
FILES=['becker_reference_check.py','becker_reference_completion.py','becker_reference_protocol.json','audit_becker_reference.py',
       'replay_becker_reference.py','deliver_becker_reference.py','BECKER_REFERENCE_CHECK.md']


def document():
    out=OUT;r=read(out/'results.json');audit=read(out/'audit.json');ex=read(out/'execution.json')
    assert audit['passed'] and not r['new_temporal_comparison_resolved'] and not ex['first_attempt_passed']
    for label in ('main','completion','audit'):shutil.copyfile(f'/tmp/ltn-becker-reference-{label}.log',out/f'{label}.log')
    sourcepaths=['runs/reconnect-becker-reference/'+p for p in ('results.json','protocol.json','source/column.npz','source/material.npz')]
    sourcepaths+=['runs/reconnect-becker-reference/reference-'+f'n{n}-k{k}.npz' for n,k in ((16384,8192),(32768,8192),(32768,16384))]
    sourcepaths+=['research/rebuild/becker.py','research/rebuild/reference.py']
    save(out/'next-gate-spec.json',{'already_executed':False,'type':'read_only_initial_material_mean_projection_interpretation',
      'reason':'reference refinement improved its empirical indicator but initial piecewise-constant projection bridges now dominate the fixed positive budget; stop automatic refinement',
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in sourcepaths],
      'scope':'derive original analytic finite-cell initial means and exact partial-cell integrals; separate fine piecewise-constant projection representation from rounding on the same saved32 material cells. Read saved grids/c0 only, no time-evolved reference or candidate generation',
      'required_outputs':['source-linked analytic integral and saved-vector residual accounting','explain what full fine-cell means determine at nonaligned coarse boundaries','retain every original bridge and both unresolved decisions','reference observation contract proposal only if supported, no new solver or acceptance change'],
      'forbidden':'no removal or reduction of old positive budget, signed cancellation for acceptance, H/threshold changes, new N/K/flow/centres, additional reference solve or candidate rerun',
      'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_software_checks':0,
      'recorded_checks_total':500,'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(out/'stage3-status.json',{'date':'2026-11-06','timezone':'Asia/Shanghai','decision':r['decision'],
      'old_case_reclassified':False,'new_reference_PDE_solves':3,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
      'new_software_checks':0,'recorded_checks_total':500,'automatic_refinement_stopped':True,'next_gate_executed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    table='\n'.join(f"| {v['cells']} | {v['steps']} | {v['projection_bridge_absolute']:.10g} | {v['initial_mean_bridge_absolute']:.10g} |" for v in r['reference_rows'])
    oldR=read(ROOT/'runs/reconnect-becker-point/reference-indicators.json')['reference_space_time_indicator_absolute'];ratio=oldR/r['reference_indicator_absolute']
    report=f'''# Becker固定参考解核验：参考变细，完整H判别仍未决

2026-11-06（Asia/Shanghai）。承接本地提交9251c17的[固定三点登记](../../runs/reconnect-becker-budget-review/next-gate-spec.json)，实现前补齐[正项计数协议](becker_reference_protocol.json)。旧README、METHOD、CODEX_TASK及所有封存科学文件保持。

**结果：{r['decision']}。** 参考空间+时间经验指标降到{r['reference_indicator_absolute']:.12g}（约改善{ratio:.4g}倍），但完整正项下H指标{r['H_indicator_unchanged_formula']:.12g}仍高于保存H下降{r['H_decrease_unchanged']:.12g}。原单例和本次新比较均未决，Stage3 incomplete，fluid ORC关闭。按停止规则不再加密，不删除初值投影差或调整H/阈值。

## 完成了什么

仅运行原独立参考.py中不变的中心面通量、时间中点Crank–Nicolson，(Nref,Kref)=(16384,8192)/(32768,8192)/(32768,16384)，δ=.08、D=.02、T=.1不变。原外物质边界零组分通量，激波不作壁。三组均通过原质量漂移1e-9和初始极值±1e-10守卫；0新候选图/传播/旧路径重跑，候选仍原32列×8行、K4/8/16。累计软件检查仍500，没有新增测试夹具。

保存细网格均值通过精确Fraction重叠积分投影到原32列，原质量/c0/σ保持。原primitive插值投影也保存作为舍入诊断。**初值桥是细单元分片常数表示投到非对齐原单元的差，包含有限单元表示误差，不能称为纯舍入。**

| Nref | Kref | 精确重叠与primitive投影差 | 投影细初值与原32列c0差 |
|---:|---:|---:|---:|
{table}

这些量均为原质量范数。协议预先固定两倍三组投影差之和、两倍三组初值差之和；没有在结果后调小倍数。

## 保持原H公式的结果

原h(b)=(2√M·max(1,max|c0|)·b+b²)/E0。预算包含原全部机器/尾项/独立路径/观察桥，替换为新的参考细化指标，并加入上述新正项：

- 参考空间差：{r['reference_space_indicator_absolute']:.12g}；参考时间差：{r['reference_time_indicator_absolute']:.12g}，两倍合计{r['positive_budget_components_absolute']['twice_refined_reference_space_time']:.12g}。
- 两倍全部初值桥：{r['positive_budget_components_absolute']['twice_all_new_initial_mean_bridges']:.12g}；两倍全部投影舍入桥：{r['positive_budget_components_absolute']['twice_all_new_projection_bridges']:.12g}。
- 总b={r['budget_absolute']:.12g}；状态下降{r['state_decrease_absolute']:.12g}已分辨；H下降{r['H_decrease_unchanged']:.12g}仍低于h(b)={r['H_indicator_unchanged_formula']:.12g}。

K16终时相对原初波动误差为{r['observations'][-1]['native_error_relative']:.12g}（旧参考为{r['observations'][-1]['old_reference_error_relative']:.12g}）。参考细化差为经验指标，不认证连续PDE误差界；单候选空间网格不证明空间收敛或移动激波T4通过。

## 首次汇总故障及审计范围

三组PDE已保存后，汇总中要求旧H逐位相等的断言失败：K4在新环境浮点复算为0.01889114413284032，原保存为0.0188911441328401，差2.22044604925e-16。原失败代码/日志和{ex['pre_completion_files_unchanged']}个补全前文件保持。追加completion仅沿用原H标量计算原H下降，另存复算差；所有H公式、正项和阈值保持，新增PDE数仍为3。

独立审计用有理数计算保存数组的精确H，并验证旧值和复算值均在原数值指标内；另用独立单次扫描精确重叠法核对全部新终态/初态投影、80/100位算术核对H公式。审计只涵盖保存端点、守卫记录和预算，不声称第二条独立PDE轨迹或严格PDE误差证书。主流程首次失败（{ex['elapsed_original_seconds']:.3f}秒），追加汇总{ex['completion_seconds']:.3f}秒；独立审计首次通过。782原文件、15附件、103份旧正式清单保持。

## 下一步和恢复

[下一项只读初值投影含义复核](../../runs/reconnect-becker-reference/next-gate-spec.json)：比较原解析初值的整单元/部分单元积分与保存细单元分片常数投影，区分表示差和舍入差，明确原物质平均对应的参考观测契约。仅作推导和保存数据复核，0新PDE/候选/网格；旧正项和两次未决均保持。未执行，不自动启动第四组参考。

保存结果复核：
```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/replay_becker_reference.py --directory runs/reconnect-becker-reference
```
该入口复算投影和预算，不求解PDE。ZIP包含研究代码/文档、782原文件、前阶段移动Becker完整结果和本轮三组参考；不是39GB全工作区，不含15附件或未取回v080。本轮仅本地交付，没有公开推送、环境发布或版本发布。
'''
    f=ROOT/'research/reconnect/BECKER_REFERENCE_CHECK.md';assert not f.exists();f.write_text(report)
    tab='\n'.join(f"{v['cells']} & {v['steps']} & {v['projection_bridge_absolute']:.5g} & {v['initial_mean_bridge_absolute']:.5g} \\\\" for v in r['reference_rows'])
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=22mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-06}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries Becker固定参考解核验}\\[5pt]{\large 参考变细，完整H判别仍未决}\\[4pt]2026年11月6日（Asia/Shanghai）\end{center}
\section*{1. 本轮结论}
参考空间与时间经验指标降到5.956372$\times10^{-9}$，约改善12.14倍。
但加上预先固定的全部初值/投影正项，H指标为9.468091$\times10^{-7}$，
仍高于原H下降5.762796$\times10^{-7}$。
\textbf{本次新比较与原单例均保留未决；Stage3 incomplete，fluid ORC关闭。}

完成原计划三组独立参考：原中心面通量和时间中点Crank--Nicolson源码不变；
$\delta=.08,D=.02,T=.1$，外物质端点零组分通量，激波不是壁。
三组均通过原质量漂移$10^{-9}$与初始极值$\pm10^{-10}$守卫。
0新候选图、0候选传播、0旧路径重跑；原32物质列$\times$8行、候选K4/8/16保持。
没有新增软件测试夹具，累计记录仍500项。

\section*{2. 新出现的主要正项是初值投影差}
\begin{center}\begin{tabular}{rrrr}\toprule
$N_{\rm ref}$&$K_{\rm ref}$&投影舍入桥&初值均值桥\\\midrule
__TABLE__
\bottomrule\end{tabular}\end{center}
表中为原质量范数。投影舍入桥对比精确Fraction重叠积分与原primitive插值；
初值均值桥对比细单元分片常数初值投影与原32列解析平均。
\textbf{后者包含有限单元表示误差，不能称为纯舍入。}
原32列质量、初值、初始波动分母始终保持。

协议在新参考求解前固定：全部三组投影桥、初值桥分别取两倍正项和。
没有看到结果后调低倍数、删除项或使用带符号抵消。
\[h(b)=\frac{2\sqrt M\max(1,\max|c_0|)\,b+b^2}{E_0}.\]
参考空间差5.939400$\times10^{-9}$、时间差1.697168$\times10^{-11}$；
两倍参考和为1.191274$\times10^{-8}$，两倍初值桥为1.614312$\times10^{-8}$，
两倍投影舍入桥为1.754962$\times10^{-13}$。
保留原全部数值和观察桥后，总$b=2.82114327\times10^{-8}$。
状态下降4.029145$\times10^{-5}$可分辨，H下降仍未超过原公式的新预算。
\newpage
\section*{3. 保存终态与参考误差}
K16相对原初波动误差为4.13061644$\times10^{-4}$，旧参考下为4.13042910$\times10^{-4}$。
这是一组候选空间网格的误差观察，不证明空间收敛或移动激波T4完成。
参考细化差是经验指标，不是认证连续PDE误差界。
按登记的停止规则，本轮到此停止，不启动第四组参考、不自动增加候选N/K。

\section*{4. 首次汇总故障与追加恢复}
三组PDE结果均已保存后，汇总程序要求旧H逐位一致的断言失败。
K4当前浮点复算比原保存H高2.22044605$\times10^{-16}$，K8/K16一致。
原失败代码、日志及21个补全前文件保持。
追加入口沿用原保存H计算原H下降，另记浮点复算差；没有更改H公式、预算或阈值，
没有重算PDE，新增PDE总数仍是3。不能把首次主流程写成成功。

独立审计以有理数计算原数组精确H，确认原保存值和当前复算值均在原数值指标内；
另用单次扫描的精确重叠积分核对全部新初态/终态投影，80/100位算术核对预算。
审计覆盖保存端点、守卫记录与预算，\textbf{不声称第二条独立PDE轨迹或严格误差证书。}
782原文件、15附件、103份旧正式清单保持，旧失败记录不改。

\section*{5. 下一项仅复核初值投影含义}
由同一解析初值计算整单元与部分单元积分，对照保存细单元分片常数投影，
区分有限单元表示差和舍入差，明确原物质质量平均对应的参考观测契约。
仅推导及保存数据复核，0新PDE/候选/网格；保留旧正项和两次未决，不以重解释追认通过。
该下一项尚未执行。Stage3仍未完整通过，fluid ORC保持关闭，$\kappa$未计算。

\section*{6. 本地交付与复核}
ZIP含研究代码/文档、782原文件、移动Becker父例与本轮三组参考、日志及审计，
不是39GB全工作区镜像，不含15附件或未取回v080。CRC/SHA与新解压复核以交付JSON为准。
本轮没有公开推送、环境或软件版本发布。
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/replay_becker_reference.py \
--directory runs/reconnect-becker-reference
\end{verbatim}
此入口只复算保存投影和预算，不调用PDE或候选求解器。
\end{document}
'''.replace('__TABLE__',tab)
    (out/f'{BASE}_report_{DATE}.tex').write_text(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n104. [固定Becker参考L形核验：完整H判别仍未决](../research/reconnect/BECKER_REFERENCE_CHECK.md)（2026-11-06）。\n\n'
      '执行已登记三组参考(16384,8192)/(32768,8192)/(32768,16384)，原参考/Becker源码不变，0候选图/作用/重跑。参考空间+时间指标5.956372e-9，但预先固定初值投影正项使原公式H指标9.468091e-7仍高于下降5.762796e-7；保持两次未决并停止加密。主汇总因旧H逐位一致断言失败（复算差2.22e-16），21文件/失败源码保持，追加汇总使用旧H并记录复算差，无额外PDE。独立精确投影/80与100位预算审计通过；原782文件、15附件、103份旧正式清单保持，累计500。下一项仅初值投影含义只读复核，Stage3 incomplete、fluid ORC关闭；仅本地交付，不发布/推送。\n')


def package():
    out=OUT;exports=ROOT/'exports';pdf=exports/f'{BASE}_report_{DATE}.pdf';archive=exports/f'{BASE}_code_results_{DATE}.zip';seal=out/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal))
    assert read(out/'audit.json')['passed'] and not read(out/'results.json')['new_temporal_comparison_resolved']
    info=subprocess.run(['pdfinfo',str(out/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           2' in info
    log=(out/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(out/pdf.name,pdf);save(out/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_visually_reviewed':True,'no_overfull_or_missing_characters':True})
    parent=ROOT/'runs/reconnect-becker-budget-review/sha256.json';entries=read(parent)
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    entries[str(parent.relative_to(ROOT))]=digest(parent)
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in out.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_becker_budget_code_results_20261104.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for directory in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file());selected.add(str(pdf.relative_to(ROOT)));selected.add(str(parent.relative_to(ROOT)))
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN fixed Becker reference check,2026-11-06. Still unresolved; Stage3 incomplete,fluid ORC OFF.
Exactly3 reference PDE solves,0 candidate graphs/actions/reruns. Original H/threshold and immutable original782 files retained.
Reference indicator5.956372e-9; added predeclared initial mean bridges leave Hbudget9.468091e-7 > saved decline5.762796e-7.
Original main FAILED after saving all3 PDE endpoints at bitwise saved-H assertion. Current K4 reduction differs2.22e-16.
Append-only completion retains saved H and reports reevaluation,with original numeric allowance. All21 precompletion files unchanged,no new PDE.
Audit verifies endpoints,independent exact projections and original budget with80/100digit arithmetic; not independent PDE trajectory replay or certified continuum accuracy.
Next only registered: read-only interpretation of initial material mean projection bridges; no automatic further reference refinement.
Contains all research/docs,782 originals,parent movingBecker cases,current3ref data,PDF/logs/audits. Not full39GB workspace,15 attachments or missingv080.
Only local delivery; no publication or GitHub push.
Saved-data replay: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/replay_becker_reference.py --directory runs/reconnect-becker-reference
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-becker-ref-restore-',dir='/tmp') as directory:
            z.extractall(directory);replay=subprocess.run([sys.executable,'research/reconnect/replay_becker_reference.py','--directory','runs/reconnect-becker-reference'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    delivery={'date':'2026-11-06','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
      'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
      'CRC_SHA_and782_originals_verified':True,'fresh_saved_data_replay_exit_code':0,'fresh_saved_data_replay_log':replay.stdout,'extracted_files_unchanged':True,
      'reference_solves':3,'candidate_solves':0,'main_first_attempt_passed':False,'append_only_completion_passed':True,'precompletion_files_preserved':21,
      'audit_passed':True,'recorded_checks_total':500,'decision':read(out/'results.json')['decision'],'stage3':'incomplete','fluid_orc_enabled':False,
      'prior_formal_manifests_verified':103,'attachments_verified_in_workspace':15,'next_gate_executed':False,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',delivery)
    (exports/f'{BASE}_downloads_{DATE}.md').write_text(f'# 固定参考核验下载\n\n[两页PDF]({pdf.name}) · [代码与结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n完整H判别仍未决，Stage3未完成，fluid ORC关闭；仅本地交付。\n')
    print(json.dumps(delivery,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['document','package']);a=p.parse_args();document() if a.mode=='document' else package()
