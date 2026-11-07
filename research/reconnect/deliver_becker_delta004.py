"""Local report and verified selected recovery package for the single delta=.04 experiment."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
OUT=ROOT/'runs/reconnect-becker-delta004'
PARENT=ROOT/'runs/reconnect-becker-delta-definition'
BASE='LTN_becker_delta004'
DATE='20261107'
FILES=['becker_delta004_sampling.py','becker_delta004_core.py','becker_delta004_study.py',
       'audit_delta004_specializations.py','audit_becker_delta004.py','becker_delta004_protocol.json',
       'BECKER_DELTA004_COMPARISON.md','deliver_becker_delta004.py']

def document():
    r=read(OUT/'results.json');a=read(OUT/'audit.json');ex=read(OUT/'execution.json');geo=read(OUT/'physical-geometry.json')
    assert a['passed'] and ex['completed']
    for name in ('main','audit'):shutil.copyfile('/tmp/ltn-delta004-'+name+'.log',OUT/(name+'.log'))
    ok=r['new_delta_temporal_qualified'];response=r['H_response_directions_consistent']
    next_type='read_only_delta004_spatial_control_definition' if ok and response else 'read_only_delta004_error_and_time_budget_review'
    next_text=('只读核对新δ单网格证据的限制，定义同一固定质量问题族上最小三档空间控制及其独立时间/H资格；只提出有预算的单项，不运行新路径或参考。'
        if ok and response else '只读定位新δ时间/H或响应未决项：逐项列出状态、H、参考与数值正项，不减旧预算、不改阈值；先判断现有数据能否区分原因，不自动增加K、网格、δ或参考。')
    gate_sources={str((OUT/name).relative_to(ROOT)):digest(OUT/name) for name in ('results.json','audit.json','execution.json','physical-geometry.json','protocol.json')}
    save(OUT/'next-gate-spec.json',{'type':next_type,'already_executed':False,'work':next_text,'sources':gate_sources,
        'max_new_graphs':0,'max_new_actual_paths':0,'max_new_independent_paths':0,'max_new_reference_solves':0,
        'new_density_sources':0,'new_software_checks':0,'recorded_checks_total':502,'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,
        'original_H_and_all_positive_terms_retained':True,'no_new_delta_three_grid_or_delta_zero_claim':True,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{key:r[key] for key in ('date','timezone','comparison_decision','new_delta_temporal_qualified',
        'H_response_directions_consistent','old_shifted_H_outcomes_preserved','old_K16_spatial_decision_unchanged',
        'new_graphs','new_actual_paths','new_independent_paths','new_reference_PDE_solves','new_density_sources',
        'new_software_checks','recorded_checks_total','new_delta_three_grid_control_completed','delta_zero_limit_claimed',
        'stage3_physical_acceptance','fluid_orc_enabled','kappa_computed')})
    t=r['time_control'];old=r['retained_old_case']['errors'];sci=lambda v:f'{v:.8e}';truth=lambda v:'已分辨' if v else '未分辨'
    labels={'native':'原生浓度','common':'共同浓度','native_H':'原生H误差','common_H':'共同H误差'}
    desc={'thinner_error_larger_resolvably':'新厚度误差增大，已分辨','thinner_error_smaller_resolvably':'新厚度误差减小，已分辨','error_change_unresolved':'误差变化未分辨'}
    rows='\n'.join(f"|{labels[k]}|{sci(old[k])}|{sci(r['errors'][k])}|{sci(r['error_changes_new_minus_old'][k])}|{sci(r['fixed_cross_indicators'][k])}|{desc[r['error_change_decisions'][k]]}|" for k in labels)
    hs='\n'.join(f"|{labels[k].replace('误差','')}|{sci(v['candidate_response'])}|{sci(v['reference_response'])}|{sci(v['response_difference'])}|{sci(v['fixed_indicator'])}|{v['direction_status']}|" for k,v in r['H_responses'].items())
    summary=f"新δ=.04的状态时间差下降{truth(t['state_resolved'])}，H时间差下降{truth(t['H_resolved'])}；合并时间资格={ok}，H响应方向一致={response}。"
    text=f'''# 固定物质质量坐标的单一δ=.04对照结果

2026-11-07（UTC）。{summary} 决定：`{r['comparison_decision']}`。Stage3 incomplete，fluid ORC关闭，κ未算。原始守卫和独立序列审计通过不代替物理验收。

## 问题与来源

严格执行BECKER_DELTA004_CONTRACT.md。固定旧μ标签、M*={geo['fixed_M_star']:.17g}、32列×8行的m和质量平均c0、D=.02、T=.1；只新增δ=.04，K8/16/32，各一条实际/完整独立路径。新56图逐张完成全部单元几何证明后封存，才加载m/c0/参考和旧侧状态。普通中心阻力不使用内部真实物质面、参考标量、F/C。旧δ=.08侧全预算保留，包括K4正项。

参考复用旧128父区间的对齐网格、质量、c0；新δ三个终态重新求解，16486/32972/32972格和8192/8192/16384步。未调用新flow.total替换M*，未用旧N32 H分支。旧v080仍未取回，不能把重建数据说成旧v080结果。

## 物质几何变化

初始外端={geo['initial_x_endpoints']}；真实单元宽度范围={geo['initial_material_cell_width_range']}；h_x/δ范围={geo['initial_hx_over_delta_range']}。固定质量不等于固定Eulerian网格/初场。B={geo['B']:.12g}，η={geo['eta']:.12g}，热导系数={geo['thermal_conductivity']:.12g}，三者随δ减半；标量D保持.02。内部真实物质面仅用于封存后的说明，未进入图构造。

## 新时间资格与独立参考指标

|项目|K8/16差|K16/32差|下降|原正项指标|判别|
|---|---:|---:|---:|---:|---|
|质量范数|{sci(t['state8_16'])}|{sci(t['state16_32'])}|{sci(t['state_decrease'])}|{sci(t['state_indicator'])}|{truth(t['state_resolved'])}|
|H|{sci(t['H8_16'])}|{sci(t['H16_32'])}|{sci(t['H_decrease'])}|{sci(t['H_indicator'])}|{truth(t['H_resolved'])}|

R1={sci(r['reference_indicators']['total_absolute'])}，B1={sci(r['B'])}，J1={sci(r['J'])}。三参考空间/时间差和全部初值、汇总舍入、质量几何桥均计入。参考细化差是数值指标，尚不是连续PDE误差的严格上界。

## 同一σ32下的终态误差变化

浓度误差按原质量范数/初始波动σ32={r['common_initial_fluctuation_norm']:.17g}定义；H误差为绝对差。G=新−旧，变薄不预设改善。跨厚度指标保留旧侧全部正项并加新侧时间及参考/数值/观测项。

|观测|旧δ=.08|新δ=.04|G|U0+U1|判别|
|---|---:|---:|---:|---:|---|
{rows}

## H响应方向与幅值

|观测|候选响应|参考响应|幅值偏差|固定指标|方向|
|---|---:|---:|---:|---:|---|
{hs}

两响应均超同一保守指标且同号，只表示响应方向一致。幅值偏差原样保留，不能据此宣布H精度通过；新δ候选空间误差没有被置零，也没有借用旧δ的三档空间通过。

## 执行、审计和保全

主流程{ex['elapsed_seconds']:.3f}秒，参考合计{ex['reference_seconds']:.3f}秒，峰值{ex['peak_RSS_MiB']:.2f}MiB；登记上限900秒/1024MiB，参考300秒。56新图，3实际/3独立路径，3参考终态；0旧路径重跑、0新密度模式、0新软件测试，累计502未改，旧502未重跑。

原始质量/极值/方差/耗散守卫通过；K32标量质量变化={sci(r['raw_K32_mass_change'])}。三个CN参考均采用原monitor策略，充分非负条件未满足步数分别为{[g['steps_without_sufficient_positivity'] for g in read(OUT/'reference/reference-runs.json')]}，原始有限性、质量和极值检查通过，不宣称无条件最大值原理。无裁剪、重归一化、拟合、阈值变化或外推修正。

独立审计{a['elapsed_seconds']:.3f}秒：全部56保存作用和完整独立序列、逐面阻力/速率/原始泊松权重、精确子物质积分和容量/舍入桥、80/100位正项比较。参考密度回调核对9个采样指纹及冻结源码；未重跑实际传播或参考PDE。782原文件、15附件、113旧正式清单及四组历史差异保持字节一致。

## 尚未通过与下一项

原δ=.08共同K32八项空间改善保持，旧共同K16空间未决保持；旧后移H为32列通过、64/128未决。新δ单N32没有三档h控制，两个δ不构成δ→0极限，同后端oracle/kernel/ledger三密度来源仍未闭合。完整Stage3仍未通过，fluid ORC关闭。

下一项尚未执行：{next_text} 本轮结束数值推进，不追加算例。仅本地报告、精选恢复ZIP和Git提交，不发布或推送。

恢复审计：`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_delta004.py --directory runs/reconnect-becker-delta004`。精选包含研究代码、文档、782原文件与选定数据，不含完整历史运行、15附件、未取回v080或完整聊天。
'''
    with (ROOT/'research/reconnect/BECKER_DELTA004_COMPARISON.md').open('x') as f:f.write(text)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write(f'\n\n114. [单一δ=.04固定物质质量对照](../research/reconnect/BECKER_DELTA004_COMPARISON.md)（2026-11-07 UTC）。\n\n{summary} 56图封存后加载标量，3实际/3独立路径与3新参考终态；参考网格/m/c0逐位复用。旧全部正项/H/阈值不变，误差增减及响应幅值如实报告；新δ无三档h或δ→0验收。保存序列、精确物质汇总和80/100位预算审计通过；782原文件、15附件、113旧正式清单保持。0新软件测试，累计502，旧502未重跑。下一项：{next_text} Stage3 incomplete，fluid ORC关闭，仅本地PDF/精选ZIP与Git恢复点，未发布/推送。\n')
    texrows='\n'.join(f"{labels[k]} & {sci(old[k])} & {sci(r['errors'][k])} & {sci(r['error_changes_new_minus_old'][k])} & {sci(r['fixed_cross_indicators'][k])} \\\\" for k in labels)
    hrows='\n'.join(f"{labels[k].replace('误差','')} & {sci(v['candidate_response'])} & {sci(v['reference_response'])} & {sci(v['response_difference'])} \\\\" for k,v in r['H_responses'].items())
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=20mm]{geometry}\usepackage{amsmath,booktabs,tabularx,hyperref,fancyhdr}
\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07 UTC}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 单一层厚度对照结果}\\[5pt]{\large 固定物质质量，Becker $\delta=.08\rightarrow.04$}\end{center}
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。} __SUMMARY__
\section*{1. 固定的问题族与实际变化}
32物质列$\times$8行，原$\mu_i,m_i,\bar c_i(0),M^*$逐位复用；$D=.02,T=.1$。
$M^*=3.6475475832251076$。$c_*(\mu)=1/2+(1/4)\cos(\pi\mu/M^*)$。
\[
\bar c_i=\frac1{m_i}\int_{\mu_i}^{\mu_{i+1}}c\,d\mu,\quad
X_{\delta,\mu}=1/\rho_\delta,\quad
\partial_t c\big|_\mu=\partial_\mu(\rho_\delta^2D\partial_\mu c).
\]
新流场生成轨迹、点密度、外物质边界及普通中心阻力图。原初始外端不同时固定：新初始端点__ENDS__，单元物理宽度范围__WIDTHS__，$h_x/\delta$范围__RATIOS__。
$B,\eta$及热导系数随$\delta$减半，标量$D$不变。内部真实物质面不进入候选构造；未用新flow.total替换原$M^*$，未复用旧N32的H。
\section*{2. 预先冻结的时间与参考控制}
只运行K8/16/32，共56图，3实际/3完整独立路径。全部图先独立核验封存再读标量。
三个新参考终态复用旧128父区间的16486/32972/32972格及初值，分别8192/8192/16384步。
新参考和全部初值、舍入、质量几何桥均计入$R_1$。
\[
B_1=2\sum_{K=8,16,32}e_K+2R_1+\text{全部横向桥},\qquad
h_m(E)=\frac{2\sqrt{\sum m}\max(1,\|c_0\|_\infty)E+E^2}{V_0}.
\]
共同预算$J_1=C_{32}B_1+$全部初值/参考/三档汇总舍入和容量缺陷项；旧侧全部预算（包括K4项）不减。
\begin{center}\begin{tabular}{lrrr}\toprule
量&时间差下降&固定正项指标&判别\\\midrule
状态范数&__STATEDEC__&__B__&__STATEOK__\\
H&__HDEC__&__HB__&__HOK__\\\bottomrule
\end{tabular}\end{center}
$R_1=$__R__，$J_1=$__J__。参考细化差是数值指标，不是已证明的连续PDE误差上界。
\section*{3. 同一初始波动分母的误差变化}
浓度误差为原质量范数除原$\sigma_{32}$，H误差为绝对差。$G=E_{.04}-E_{.08}$；$U=U_0+U_1$保留全部正项。变薄不预设精度提高。
\begin{center}\small\begin{tabular}{lrrrr}\toprule
观测&旧$\delta=.08$&新$\delta=.04$&$G$&$U$\\\midrule
__ERRORROWS__
\bottomrule\end{tabular}\end{center}
__DECISIONS__
\newpage
\section*{4. H响应：方向和幅值分开报告}
\begin{center}\small\begin{tabular}{lrrr}\toprule
观测&候选$\Delta H$&参考$\Delta H$&候选减参考\\\midrule
__HROWS__
\bottomrule\end{tabular}\end{center}
__HSTATUS__。候选和参考响应各自超同一保守指标且同号，仅表示方向一致；幅值偏差不能用来拟合容许阈值，方向一致不等于H精度验收。新$\delta$没有候选三档空间控制，不能将其空间误差当作零，也不能继承旧$\delta$空间通过。
\section*{5. 原始守卫、独立审计与执行证据}
主流程__SECONDS__秒，参考合计__REFSECONDS__秒，峰值__RSS__MiB，均在登记900秒/1024MiB、参考300秒内。56新图、3实际/3独立路径、3参考终态；0旧重跑、0新密度模式、0新软件测试，累计502未改，旧502未重跑。
原始质量/极值/方差/耗散及完整独立序列检查通过，K32标量质量变化__MASS__。三个参考CN充分非负条件未满足步数__CN__；monitor策略下原始有限性、质量及极值守卫通过，不宣称无条件最大值原理。未裁剪、重归一化、拟合F/C、外推修正或修改阈值。
独立审计__AUDITSECONDS__秒：逐面阻力、速率、原始泊松权重和全部56保存作用，完整独立序列，精确子物质积分、容量/舍入桥及80/100位正项比较。核验参考回调9个密度采样指纹与冻结源码；审计不重跑实际指数传播或参考PDE。
782原文件、15附件、113旧正式清单及四组历史差异保持。源码、协议和来源在新数据前锁定；本轮实际结果及审计逐项保存，可用精选包重放当前只读审计。
\section*{6. 当前边界与下一项}
旧$\delta=.08$共同K32八项空间改善保持，旧共同K16空间仍未决；旧后移H仍为32列通过、64/128未决。新$\delta$单网格不等于三档h控制，两个$\delta$不是$\delta\to0$极限；同后端oracle/kernel/ledger三密度来源仍未闭合。完整Stage3尚未通过，fluid ORC继续关闭。
__NEXT__
\section*{7. 本地交付和恢复}
附代码、原始782文件与精选运行数据ZIP，含结果JSON、逐步审计、来源哈希和本报告源码。
可重放audit\_becker\_delta004.py；依赖NumPy/SciPy，固定BLAS/OMP各4线程。先核验review-manifest.json和原文件哈希。
精选包不是完整工作区，不含15附件、未取回v080、全部历史运行或完整聊天。旧v080不能声称已恢复。本轮仅本地PDF、精选ZIP和Git恢复点，未发布或推送。
\end{document}
'''
    fields={'SUMMARY':summary.replace('_',r'\_'),'ENDS':str([round(x,10) for x in geo['initial_x_endpoints']]),
        'WIDTHS':str([round(x,10) for x in geo['initial_material_cell_width_range']]),'RATIOS':str([round(x,8) for x in geo['initial_hx_over_delta_range']]),
        'STATEDEC':sci(t['state_decrease']),'B':sci(r['B']),'STATEOK':truth(t['state_resolved']),
        'HDEC':sci(t['H_decrease']),'HB':sci(t['H_indicator']),'HOK':truth(t['H_resolved']),
        'R':sci(r['reference_indicators']['total_absolute']),'J':sci(r['J']),'ERRORROWS':texrows,'HROWS':hrows,
        'DECISIONS':'；'.join(labels[k]+'：'+desc[r['error_change_decisions'][k]] for k in labels)+'。',
        'HSTATUS':'；'.join(labels[k].replace('误差','')+('响应方向一致' if v['direction_status']=='direction_consistent' else '响应方向未决或不一致') for k,v in r['H_responses'].items()),
        'SECONDS':f"{ex['elapsed_seconds']:.3f}",'REFSECONDS':f"{ex['reference_seconds']:.3f}",'RSS':f"{ex['peak_RSS_MiB']:.2f}",
        'MASS':sci(r['raw_K32_mass_change']),'CN':str([g['steps_without_sufficient_positivity'] for g in read(OUT/'reference/reference-runs.json')]),
        'AUDITSECONDS':f"{a['elapsed_seconds']:.3f}",'NEXT':'下一项尚未执行：'+next_text+'本轮停止追加数值算例。'}
    for key,value in fields.items():tex=tex.replace('__'+key+'__',value)
    tex=re.sub(r'(?<![0-9A-Za-z_.])(-?\d+(?:\.\d+)?)[eE]([+-]\d+)',lambda m:r'\ensuremath{'+m[1]+r'\!\times\!10^{'+str(int(m[2]))+'}}',tex)
    assert '__' not in tex
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)


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
    with zipfile.ZipFile(exports / 'LTN_becker_delta_definition_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/becker_delta004_protocol.json')['source_files'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    note = """LTN fixed-material delta=.04 single-case result,2026-11-07 UTC.
Read BECKER_DELTA004_COMPARISON.md and results/audit.json for actual outcomes and unresolved gates.
Fixed old labels,Mstar,m,c0;32columns,8rows,K8/16/32;56 new graphs,3actual/3independent and3new reference terminals.
All56 independent geometry proofs sealed before scalar loading. Old reference mesh/c0 reused bitwise;new reference terminals solved.
Old positive terms/H formula/thresholds retained;no clipping,renormalization,F/C fit.
No new-delta three-grid spatial acceptance or delta->0 claim;H response direction is not magnitude accuracy acceptance.
782 originals,research/docs,selected records only;not full workspace,15 attachments,unrecoveredv080 or chat.
Verify review-manifest.json/original-handoff-sha256.json. Replay read-only audit with NumPy/SciPy:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_delta004.py --directory runs/reconnect-becker-delta004
Stage3 incomplete,fluid ORC OFF;recorded tests502,not rerun. Local only,no publication/push.
"""
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
        with tempfile.TemporaryDirectory(prefix='ltn-delta004-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_becker_delta004.py',
                '--directory', 'runs/reconnect-becker-delta004'], cwd=directory,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 54, 'prior_formal_manifests_verified': 113,
        'attachments_verified_in_workspace': 15, 'new_graphs': 56, 'new_actual_paths': 3, 'new_independent_paths': 3, 'new_reference_PDE_solves': 3,
        'new_software_checks': 0, 'recorded_checks_total': 502, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# 单一层厚度对照结果下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。本轮δ=.04单例已执行；后续门尚未执行；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['document','package'])
    args=p.parse_args()
    document() if args.mode=='document' else package()
