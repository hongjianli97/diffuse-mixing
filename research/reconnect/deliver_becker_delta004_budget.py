"""Local report and selected recovery package for the saved delta=.04 H-budget review."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
OUT=ROOT/'runs/reconnect-becker-delta004-budget'
PARENT=ROOT/'runs/reconnect-becker-delta004'
BASE='LTN_becker_delta004_budget'
DATE='20261107'
FILES=['becker_delta004_budget_review.py','audit_becker_delta004_budget.py','becker_delta004_budget_protocol.json',
       'BECKER_DELTA004_BUDGET_REVIEW.md','deliver_becker_delta004_budget.py']

def document():
    r=read(OUT/'results.json');ex=read(OUT/'execution.json');audit=read(OUT/'audit.json');old=read(OUT/'source/results.json')
    assert audit['passed'] and ex['completed'] and all(r['conditional_obstruction_proofs'].values())
    for name in ('main','audit'):shutil.copyfile('/tmp/ltn-delta004-budget-'+name+'.log',OUT/(name+'.log'))
    sources={str((OUT/name).relative_to(ROOT)):digest(OUT/name) for name in ('results.json','audit.json','protocol.json')}
    names=['docs/METHOD.md','docs/CODEX_TASK.md','research/reconnect/T1_UNIFORM_COMPARISON.md','research/reconnect/T1_UNIFORM_CONTROL.md',
        'research/reconnect/t1_uniform_protocol.json','research/reconnect/t1_uniform_core.py','runs/reconnect-t1-uniform/results.json',
        'runs/reconnect-t1-uniform/numerical-audit.json','runs/reconnect-t1-uniform/n32/material-reference.npz','runs/reconnect-t1-uniform/builder-inputs.npz']
    seal=read(PARENT/'sha256.json')
    for name in names:
        assert digest(ROOT/name)==seal[name];sources[name]=seal[name]
    assert read(ROOT/'runs/reconnect-t1-uniform/results.json')['T1_registered_case_passed']
    assert read(ROOT/'runs/reconnect-t1-uniform/numerical-audit.json')['T1_registered_case_passed']
    next_text='只读定义已通过oracle控制的T1均匀周期平滑基准上的同后端oracle/kernel/ledger最小密度来源标定；固定已存N32的m/c0/D/几何和参考，核查各来源所需输入、无效密度拒绝和禁用F修正，给出至多一个后续算例的冻结预算。不得计算新密度数组/图/状态，不重跑旧路径。均匀标定不代表变密度、激波或完整Stage3验收；Becker H与空间缺口保留。'
    save(OUT/'next-gate-spec.json',{'type':'read_only_smooth_T1_same_backend_density_control_definition','already_executed':False,
        'work':next_text,'sources':dict(sorted(sources.items())),'existing_oracle_case':'T1_static_uniform_material_mean_control_passed',
        'candidate_columns_per_axis':32,'density_modes_to_define':['oracle','particle_reconstruction','ledger_experimental'],
        'same_mass_initial_mean_D_geometry_boundary_and_exchange_backend_required':True,'F_correction_enabled':False,
        'no_new_density_arrays_or_flows':True,'max_new_graphs':0,'max_new_actual_paths':0,'max_new_independent_paths':0,'max_new_reference_solves':0,
        'new_software_checks':0,'recorded_checks_total':502,'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,
        'original_Becker_failures_not_waived':True,'full_density_validation_claim_allowed':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-07','timezone':'Etc/UTC','budget_review_completed':True,
        'delta004_time_qualified':False,'delta004_H_response_direction_consistent':True,'conditional_obstructions':r['conditional_obstruction_proofs'],
        'new_graphs_or_paths_or_reference_solves_or_density_arrays':0,'new_software_checks':0,'recorded_checks_total':502,
        'next_density_contract_defined_or_run':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    groups=r['positive_B_groups'];v=r['precision_checks'][-1];t=r['saved_time_control'];sci=lambda x:f'{float(x):.8e}'
    labels={'machine_allowance':'原机器余量','raw_tail':'原始尾项','measured_sequence_difference':'实测完整序列差',
        'reference_space':'参考空间差','reference_time':'参考时间差','reference_initial_rounding_geometry_bridges':'参考初值/汇总/几何桥','transverse_bridges':'横向观测桥'}
    rows='\n'.join(f"|{labels[k]}|{sci(o['float'])}|{100*r['budget_shares'][k]:.8f}%|" for k,o in groups.items())
    identity='\n'.join(f"|{x['from_K']}→{x['to_K']}|{sci(x['linear_positive_sum_over_V0']['float'])}|{sci(x['linear_negative_sum_over_V0']['float'])}|{sci(x['linear_net_over_V0']['float'])}|{sci(x['quadratic_over_V0']['float'])}|{x['spatial_linear_cancellation_ratio']['float']:.2f}|" for x in r['H_variance_change_identity'])
    text=f'''# δ=.04的H时间预算与保存状态复核

2026-11-07（UTC）。只读复核完成，0新物理数据/图/路径/参考/密度/软件测试，累计502不变。旧判别`{r['parent_decision_unchanged']}`原样保留：状态时间分辨，H时间未分辨，H响应方向一致不等于幅值精度通过。Stage3 incomplete，fluid ORC关闭，κ未算。

## 结论及适用范围

参考空间与时间项共占B的{100*(r['budget_shares']['reference_space']+r['budget_shares']['reference_time']):.6f}%，原机器余量占{100*r['budget_shares']['machine_allowance']:.6f}%。但是只细化参考不能解除当前H未决：保留机器余量产生的H下界={sci(v['machine_only_lower_bound'])}，已高于当前H差下降={sci(t['H_decrease'])}。这个量是原保守数值余量，不是实测浮点误差。

同样，只收紧机器余量而保留参考项也不够，参考项单独产生的H下界={sci(v['reference_only_lower_bound'])}。这些都是单调多项式的条件下界论证，不是在正式判据中删除任何正项；不建议借此修改阈值或误差公式。

## 全部正项计账

B={sci(old['B'])}，原H指标={sci(t['H_indicator'])}。表内均为展开后的非负项；汇总浮点舍入单独记录为有符号余差，并非被扣除的误差项。

|来源|进入B的绝对量|占原B比例|
|---|---:|---:|
{rows}

原B精确二进制值减上述有理数正项和的余差为{r['saved_accumulation_residual_signed']['exact']}。全部正项加该记账余差严格重组原B。H多项式h(E)=(A E+E²)/V0未改：按各组展示线性A E/V0项，整个B²/V0={sci(v['full_H_quadratic_term'])}单列，不把非线性交叉项武断地归为某个物理误差来源。

原公式反解当前观测下降对应的B尺度约{sci(v['critical_B_for_saved_gap_diagnostic_only'])}，仅作诊断，不成为替换阈值或拟合目标。

## 为什么机械后移时间档也无效

若未来仍保留本轮全部正项，则B_new≥B，h(B_new)≥h(B)。对于K16/32/64，任何未知H64均满足：

`|H16−H32|−|H32−H64| ≤ |H16−H32| = {sci(t['H16_32'])} < h(B) = {sci(t['H_indicator'])}`。

因此这项后移下降判据在该保留政策下不可能通过，即使新增H差恰为零。结论只针对该固定旧预算与后移三元组，不证明所有时间离散方法、其他物理问题或新空间网格都无法收敛。本轮没有计算K64，也不自动追加K、网格、δ或参考。

## H小变化的精确状态恒等式

用保存binary64状态的精确有理数值，令w=v−u、u_c=u−mean_m(u)、w_c=w−mean_m(w)，则

`H(v)−H(u) = −[2<u_c,w_c>_m + ||w_c||_m²]/V0`。

|时间档|线性正贡献和/V0|线性负贡献和/V0|线性净和/V0|二次项/V0|线性绝对和/净和|
|---|---:|---:|---:|---:|---:|
{identity}

空间单元的正负线性贡献明显抵消；线性净项与二次项在这两对均同号，不能误称线性项与二次项相互抵消。H是全局方差观测，状态差的下降不能直接代替H差的下降。

精确状态H差下降={sci(v['exact_state_H_decrease'])}，原浮点H差下降={sci(t['H_decrease'])}；区别约1.10e−16，远小于原H指标。重新以高精度评价已存H并不会把本项改为通过。此抵消是保存状态的数学关系，不说明浮点舍入造成了当前物理误差。

## 不能由现有数据得出的结论

旧δ=.08与新δ=.04浓度误差分别约4.06e−4/7.05e−4，四项误差变化已按原指标分辨；它们不能把误差进一步因果分配给点密度、阻力、几何或层厚分辨率。新δ初始h_x/δ约1.335–1.583，尚无三档h控制。候选和参考H响应方向一致，不保证响应幅值准确；参考细化差不是认证连续PDE误差上界。

所有旧H公式、阈值、预算及失败保持。旧共同K16空间与64/128后移H未决，新δ空间、δ→0和同后端三密度闭合均未完成。未对时间阶作拟合，未外推状态，未裁剪、归一化或拟合F/C。

## 独立审计、保全与下一项

主复核{ex['elapsed_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB，在120秒/1024MiB内。独立审计{audit['elapsed_seconds']:.3f}秒：精确二进制预算重组、独立二阶矩方差恒等式及80/100位下界比较。无流场/几何/PDE求解器导入或调用。782原文件、15附件、114旧正式清单及四组历史差异保持。

停止这条已证无效的参考单项细化或机械后移补算路线。下一项尚未执行：{next_text} 这是继续已通过oracle的平滑基准的独立缺口，不是绕过Becker门或新增通过结论。

恢复：`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_delta004_budget.py --directory runs/reconnect-becker-delta004-budget`。

仅本地报告、精选ZIP和Git恢复点，不发布/推送。精选包不含全部历史运行、15原附件、未取回v080或完整聊天；完整保全仍需原工作区。
'''
    with (ROOT/'research/reconnect/BECKER_DELTA004_BUDGET_REVIEW.md').open('x') as f:f.write(text)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n115. [δ=.04 H时间预算只读复核](../research/reconnect/BECKER_DELTA004_BUDGET_REVIEW.md)（2026-11-07 UTC）。\n\n'
        '参考空间/时间项占B约87.217%，但机器项H下界7.9587e−9已高于H下降4.3935e−9；只细化参考不能解锁。保留全B时，后移K16/32/64下降上限3.0348e−9低于原指标6.2263e−8，不能通过。精确保存状态方差恒等式显示空间线性贡献抵消，高精度H差仍未决；不把余量等同实测浮点误差、不作物理误差因果分配。0新图/路径/参考/密度/软件测试，累计502；独立有理数和80/100位审计通过，782原文件/15附件/114旧正式清单保持。下一项仅定义已通过oracle的T1均匀平滑基准同后端三密度标定契约，不产生新数组或关闭Becker缺口。Stage3 incomplete，fluid ORC关闭；仅本地PDF/精选ZIP和Git恢复点，未发布/推送。\n')
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,tabularx,hyperref,fancyhdr}
\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07 UTC}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries H时间未决：预算与状态复核}\\[5pt]{\large 只读保存的Becker $\delta=.04$单例}\end{center}
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。}
本轮未计算新点、图、密度、状态或参考。原H时间未通过及四项误差增大结论保留。新增证据解释该判据为何不能靠单项参考细化或机械后移时间档解锁，不是新的物理验收。
\section*{1. 参考占主要预算，但单独细化仍不足}
原状态时间差下降已分辨；H下降$G_H=$__GAP__，原指标$h(B)=$__FULL__。
\[
 h(E)=\frac{A E+E^2}{V_0},\qquad A=2\sqrt{\sum m}\max(1,\|c_0\|_\infty),\qquad B=__B__.
\]
\begin{center}\begin{tabular}{lrr}\toprule
进入B的正项组&绝对量&占B比例\\\midrule
__BROWS__
\bottomrule\end{tabular}\end{center}
参考空间/时间共占87.217\%，机器余量占12.782\%。机器余量是原保守估计，并非实测浮点误差；完整独立序列差另列。
全部有理数正项加单独记录的浮点汇总余差，精确重组原B。H线性项按来源列出，完整$B^2/V_0=$__QUAD__单列，不将非线性交叉项当作可独立分配的物理误差。
\section*{2. 三个有条件的下界结论}
\begin{tabularx}{\textwidth}{p{37mm}X}\toprule
保留条件&结论\\\midrule
只改善参考；保存路径与原机器余量不变&$h(B_{\rm machine})=$__MFLOOR__$>G_H$。因此参考差即使降为零，也不能通过当前H检查。\\
只改善机器余量；保存参考项不变&$h(B_{\rm reference})=$__RFLOOR__$>G_H$。单独改善这一项仍不够。\\
后移K16/32/64；保留全部旧预算&任何未知$H_{64}$均有$|H_{16}-H_{32}|-|H_{32}-H_{64}|\le$__CEILING__$<h(B)$。\\\bottomrule
\end{tabularx}
这些是假定保留条件下的数学下界；正式预算没有删除任何项、没有改阈值。后移结论只适用于该三元组及保留政策，不能外推为所有网格/方法均不收敛。未运行K64，不追加扫描。
\newpage
\section*{3. H小变化来自怎样的保存状态关系}
对保存binary64状态采用精确有理数算术。记$w=v-u$，下标c表示减去质量平均，则
\[
 H(v)-H(u)=-\frac{2\langle u_c,w_c\rangle_m+\|w_c\|_m^2}{V_0}.
\]
独立审计用二阶矩$\sum m u^2-(\sum m u)^2/\sum m$核验方差，不复用主代码的中心化平方公式。
\begin{center}\small\begin{tabular}{lrrr}\toprule
时间档&线性净项/$V_0$&二次项/$V_0$&线性绝对和/净和\\\midrule
__IROWS__
\bottomrule\end{tabular}\end{center}
不同空间单元的正负线性贡献明显抵消；这两对的线性净项与二次项同号，不能误称两者相互抵消。
H是全局方差观测，状态范数的收敛迹象不能代替H判据。

精确状态算得H差下降__EXACTGAP__，原浮点值__GAP__，相差约$1.10\times10^{-16}$，都低于原指标。提高已存H的评价精度不会改变本项结论。空间抵消是状态的数学关系，不证明浮点舍入造成了物理误差。
\section*{4. 现有数据不能归因的部分}
厚度从.08减到.04后，原生相对浓度误差约$4.06\times10^{-4}\to7.05\times10^{-4}$，四项误差变化已按原指标分辨；新$h_x/\delta$约1.335--1.583。但是两厚度、单一新网格不能把误差因果分配给点密度、阻力、几何或厚度分辨率。候选与参考H响应方向一致，不等于幅值精度通过。
新$\delta$三档h控制及$\delta\to0$结论均缺，参考细化差不是认证连续PDE误差界。旧共同K16空间、旧64/128后移H未决均保留。没有拟合时间阶、外推状态、裁剪、归一化、F/C拟合或修改物理阈值。
\section*{5. 下一项：平滑基准的密度来源契约}
停止已证无效的上述补算路线。下一项仅只读定义已通过oracle控制的T1均匀周期平滑基准上的同后端oracle/kernel/ledger最小标定。
固定已存N32的m、质量平均c0、D、几何和参考；检查密度来源各自所需输入、无效区域拒绝、禁用F修正，最多提出一个冻结预算的后续算例。该契约尚未定义，本轮不产生新密度数组、图或状态。
均匀标定只能检查常量一致性和输入接口，不能替代变密度、激波或完整三来源验收；所有Becker缺口继续保留。Stage3仍incomplete，fluid ORC关闭。
\section*{6. 核验与交付}
主复核__SECONDS__秒、__RSS__MiB；独立审计__AUDITSECONDS__秒，精确预算重组与80/100位下界核验通过。0新物理计算、0新软件测试，累计502未改，旧502未重跑。782原文件、15附件、114旧正式清单及四组历史差异保持。
代码、证据、报告源码与精选结果ZIP可重放audit\_becker\_delta004\_budget.py。
精选包不是完整工作区，不含15附件、未取回v080、全部历史运行或聊天。仅本地保存，未发布或推送。
\end{document}
'''
    fields={'GAP':sci(t['H_decrease']),'FULL':sci(t['H_indicator']),'B':sci(old['B']),
        'BROWS':'\n'.join(f"{labels[k]} & {sci(o['float'])} & {100*r['budget_shares'][k]:.6f}\\% \\\\" for k,o in groups.items()),
        'QUAD':sci(v['full_H_quadratic_term']),'MFLOOR':sci(v['machine_only_lower_bound']),'RFLOOR':sci(v['reference_only_lower_bound']),
        'CEILING':sci(t['H16_32']),'EXACTGAP':sci(v['exact_state_H_decrease']),
        'IROWS':'\n'.join(f"{x['from_K']}→{x['to_K']} & {sci(x['linear_net_over_V0']['float'])} & {sci(x['quadratic_over_V0']['float'])} & {x['spatial_linear_cancellation_ratio']['float']:.2f} \\\\" for x in r['H_variance_change_identity']),
        'SECONDS':f"{ex['elapsed_seconds']:.3f}",'RSS':f"{ex['peak_RSS_MiB']:.2f}",'AUDITSECONDS':f"{audit['elapsed_seconds']:.3f}"}
    for key,value in fields.items():tex=tex.replace('__'+key+'__',value)
    # Use \ensuremath so both text and formula placeholders retain correct signed exponents.
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
    with zipfile.ZipFile(exports / 'LTN_becker_delta004_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/becker_delta004_budget_protocol.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    selected.update(read(OUT/'next-gate-spec.json')['sources'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note = """LTN delta=.04 H budget review,2026-11-07 UTC.
Read BECKER_DELTA004_BUDGET_REVIEW.md. No new points,graphs,density,states,PDE or software tests.
Original failed H gate and all positive terms retained. Conditional floor proofs do not change acceptance.
Exact saved-state variance identities distinguish spatial cancellation from measured roundoff or physical error causation.
Next only define a smooth T1 density-source calibration contract;NOT executed;does not close Becker/shock gaps.
782 originals,research/docs,selected records only;not full workspace,15 attachments,unrecoveredv080 or chat.
Verify review-manifest.json/original-handoff-sha256.json. Replay current read-only audit:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_delta004_budget.py --directory runs/reconnect-becker-delta004-budget
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
        with tempfile.TemporaryDirectory(prefix='ltn-delta004-budget-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_becker_delta004_budget.py',
                '--directory', 'runs/reconnect-becker-delta004-budget'], cwd=directory,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 21, 'prior_formal_manifests_verified': 114,
        'attachments_verified_in_workspace': 15, 'new_graphs': 0, 'new_actual_paths': 0, 'new_independent_paths': 0, 'new_reference_PDE_solves': 0,
        'new_software_checks': 0, 'recorded_checks_total': 502, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# H时间预算复核下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。本轮仅只读复核；后续平滑流密度来源契约尚未定义；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['document','package'])
    args=p.parse_args()
    document() if args.mode=='document' else package()
