"""Local report and selected recovery package for the smooth T1 density-source design."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
OUT=ROOT/'runs/reconnect-t1-density-definition'
PARENT=ROOT/'runs/reconnect-becker-delta004-budget'
BASE='LTN_t1_density_definition'
DATE='20261107'
FILES=['t1_density_definition.py','audit_t1_density_definition.py','t1_density_definition_protocol.json',
       'T1_DENSITY_SOURCE_CONTRACT.md','T1_DENSITY_SOURCE_DEFINITION.md','deliver_t1_density_definition.py']

def document():
    p=read(ROOT/'research/reconnect/t1_density_definition_protocol.json');d=read(OUT/'definition.json')
    inv=read(OUT/'input-inventory.json');a=read(OUT/'audit.json');ex=read(OUT/'execution.json')
    assert a['passed'] and ex['completed'] and not d['new_density_arrays_or_coefficients_generated']
    for name in ('main','audit'):shutil.copyfile('/tmp/ltn-t1-density-definition-'+name+'.log',OUT/(name+'.log'))
    sources=dict(p['sources'])
    for name in ('definition.json','input-inventory.json','symbolic-checks.json','audit.json'):
        sources[str((OUT/name).relative_to(ROOT))]=digest(OUT/name)
    save(OUT/'next-gate-spec.json',{'type':'single_static_T1_N32_density_calibration_kernel3_paths_ledger_identity_reuse',
        **d['proposed_case'],'already_executed':False,'sources':dict(sorted(sources.items())),
        'contract':p['contract'],'contract_sha256':digest(ROOT/p['contract']),
        'implementation_and_audit_code_lock_required_before_new_density_or_test_arrays':True,
        'ledger_reuse_failure_stops_without_assigning_rho_or_g':True,'all_raw_and_independent_guards_unchanged':True,
        'static_T1_time_semigroup_rule_not_Becker_H_waiver':True,'report_density_bias_without_physical_accuracy_threshold':True,
        'acceptance_scope':'implementation controls and classified source biases only; no general density/shock/Stage3 pass',
        'stop':'record software/raw guard/identity/control failure or cap; no automatic N,K,h,reference or model expansion; adverse physical bias is reported,not tuned away',
        'current_recorded_checks_total':502,'all_Becker_unresolved_gates_preserved':True,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-07','timezone':'Etc/UTC','density_source_contract_defined':True,
        'new_case_executed':False,'new_density_fields_or_coefficient_packets_or_paths_or_PDE':0,'new_software_checks':0,
        'recorded_checks_total':502,'future_tests_if_both_pass':504,'original_Becker_failures_preserved':True,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    md=f'''# 平滑T1三密度来源：定义与可行性核验

2026-11-07（UTC）。已完成[冻结契约](T1_DENSITY_SOURCE_CONTRACT.md)和独立只读审计；**尚未计算任何新密度、系数或路径**。本轮累计软件检查仍502，Stage3 incomplete，fluid ORC关闭，κ未算。

## 固定一个可追溯的已通过oracle基准

复用旧T1单位二维周期域，N=32（1024节点）、2048共享面、D=.02、T=.05、K4/8/16，观察t=0/.025/.05。原m=1/1024、质量平均c0、解析参考、IDs/位置、所有面和oracle三条实际/完整独立路径均核对。源数据仍是旧实验，不重新标为新结果。

34个来源哈希、旧几何审计回执和路径初始/步长核对通过。m/c0与参考原样复用；浓度来源比较用原T1共同分母sigma16={inv['saved_original_common_sigma16']:.17g}，本档分母另报。质量只在读取mass键后给核重构，不将c0/参考传入密度或系数构造。

## 三来源的实现选择

|来源|允许的信息与规则|本例作用与限制|
|---|---|---|
|oracle|原给定rho=1及封存g/路径|作为旧基线，不重跑|
|particle_reconstruction|原位置/ID/周期/m；原二维径向核，固定h=1/16，含自项|核只重构rho，交换仍普通中心阻力；不离散重归一化、不扫描h|
|ledger_experimental|给定rho0，固定周期邻居环的有向面积比|新的最小实验适配器；不执行旧v3/v4，不做F拟合、修补、重锚或回退|

核 W=9/(4πh²)(1−q/2)^6(1+3q+35q²/12)，q=r/h<2；支撑外0。连续积分的径向部分为2/9，独立beta积分也给出2/9，乘角向因子9/2得1。这个单位积分不保证有限粒子求和等于1。均匀格的45项有限和rho*保留符号表达，本轮没有数值求值；未来必须报告原始常量偏差，不能填入1使其通过。

ledger采用固定四向邻居环，Astar=1/2Σcross(v_j,v_next)，rhoL=rho0*Astar(0)/Astar(t)。使用精确有理有向面积，非正/自交/退化直接拒绝，不取abs。原静止格的菱形面积Astar=2/N²，而m=1/N²，因此rho0*Astar=2m；这是重叠代理环，不是物质单元。绝不替换原m或除2“修正”锚质量。静止面积比1仅给出本例身份关系，不验证运动/压缩密度。

## 为什么不直接运行旧ledger

原lhdx._star使用绝对三角面积，chained_lnJ_v2有重锚和非有限增量沿用上一值；ledger_ub包含最小二乘F/混合张量路径；ledger_v4含F拟合、弧选择与修补。8段带行号/哈希的源码摘录原样保存，作为设计依据，未导入或执行这些驱动。新固定锚点适配器保留面积比思想，不能被描述为旧v4已经无梯度验证。

## 冻结的最小后续执行

两个新静态密度字段、两个新普通半阻力系数包，沿用封存几何。独立检查每条共享面以及原m/D/几何字段不变，两包全部封存后才加载标量、参考和旧路径。

ledger的2048条有理g必须与旧oracle逐项完全相同；同m/c0/生成元/时间序列下可直接引用旧三条路径，明确标注基于系数身份复用。若不相同，保存差异后停止，不赋rho=1、不覆盖g、不自动补ledger求解。

仅kernel新增3实际/3完整独立路径（K4/8/16、共28作用区间），0新几何、0参考PDE、0旧路径重跑。先冻结实现与审计，再通过两个必要软件检查组（输入与拒绝；系数接口），通过后才将502记为504。当前未执行这些检查。软件上限120秒，科学上限300秒/1024MiB。

## 时间、符号及物理偏差的分栏规则

静态生成元的指数半群与时间分割无关；沿用原T1“时间差在原数值指标内”的控制，不要求近零差呈下降。新H时间差也受原h_m(E)控制。这不修改Becker时变H下降规则，原未决继续保留。

封存系数后，才用独立80/100位rho*和均匀离散符号lambda_s=rho*_s·8DN²sin²(π/N)检查实际完整传播；不从测得衰减、平均rho或g拟合lambda，不用符号替代实际状态。常量/特征模作用残差先过原1e−10代数守卫；原解析评价余量、机器/尾项、完整序列差和残差时间项均保留。

来源误差比较按契约逐项相加：双方K4/8及K8/16状态差，双方全部K4/8/16数值正项，4倍原参考评价余量，2倍初始桥和双方残差时间项。原生浓度变化与B/sigma16比较，H误差变化与原h_m(B)比较；增大、减小、未分辨均允许，不能按结果换h/阈值。密度物理偏差单列，代数1e−10不是新的密度物理精度标准。

最终只报告实现控制和偏差分类。单一静止均匀标定无法验证密度梯度、运动、压缩、接触/激波或三来源空间收敛，也无法区分恒D和恒Gamma；不因其结果豁免任何Becker缺口或开启ORC。

## 执行记录与交付范围

主定义核验{ex['elapsed_seconds']:.3f}秒、峰值{ex['peak_RSS_MiB']:.2f}MiB；独立审计{a['elapsed_seconds']:.3f}秒。0新rho/系数/图/状态/PDE，0新增软件测试，旧502未重跑。782原文件、15附件、115旧正式清单及四组历史差异保持。

下一项已登记但未执行：[next-gate-spec.json](../../runs/reconnect-t1-density-definition/next-gate-spec.json)。新增计算驱动、源适配器、两个软件检查和独立审计必须在任何新测试/密度数组前锁定。失败或超限保存停止，不自动扩大N/K/h/模型/参考。

恢复审计：`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_t1_density_definition.py --directory runs/reconnect-t1-density-definition`。

仅本地PDF、代码/精选结果ZIP和Git恢复点，不发布/推送、不生成新版本。精选包不是完整工作区，不含15附件、全部历史运行、未取回v080或完整聊天；完整历史保全仍需原工作区。
'''
    with (ROOT/'research/reconnect/T1_DENSITY_SOURCE_DEFINITION.md').open('x') as f:f.write(md)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n116. [平滑T1三密度来源契约定义](../research/reconnect/T1_DENSITY_SOURCE_DEFINITION.md)（2026-11-07 UTC）。\n\n'
        '复用已通过oracle的N32静态周期T1，34来源哈希、旧1024节点/2048面/m/c0/参考和三条旧路径核对；原二维径向核固定h=1/16，不离散归一化，45项rho*仅符号未求值；新固定有向环ledger代理锚质量为2m，不能替换m，不执行旧F拟合/修补/回退驱动。下一项仅2字段/2系数包，ledger须通过逐项系数身份后引用旧oracle路径；只新增kernel的3实际/3独立路径，28区间，0新几何/参考PDE。两项软件组未来通过才502→504，当前仍502；当前0新密度/系数/路径。静态半群时间控制沿用原T1，不豁免Becker；物理偏差与实现控制分列。独立符号积分/来源/旧输入审计通过，782原文件/15附件/115旧正式清单保持。Stage3 incomplete，fluid ORC关闭，仅本地PDF/精选ZIP和Git恢复点，未发布/推送。\n')
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,tabularx,hyperref,fancyhdr}
\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07 UTC}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 平滑T1三密度来源契约}\\[5pt]{\large 设计与可行性核验完成，单例尚未执行}\end{center}
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。}
本轮0新密度、系数、图、状态或PDE；所有Becker未决保留。不把静止均匀标定当作一般变密度或激波验收。
\section*{1. 固定已通过oracle的同后端基准}
单位二维周期域、单位厚度，N32（1024节点）、2048共享面，$D=.02,T=.05$，K4/8/16；观察$t=0,.025,.05$。
旧IDs、位置、真实$m=1/1024$、质量平均$c_0$、参考及oracle三条实际/完整独立路径均按哈希复用。所有来源用同一普通Voronoi半阻力交换；核只估计密度。34份来源核对，8段历史源码摘录带行号与哈希保存。
\par\medskip
\begin{tabularx}{\textwidth}{p{36mm}XX}\toprule
来源&允许输入&限制\\\midrule
oracle&原给定点密度1、旧系数和路径&不重跑旧路径\\
粒子核重构&原位置/ID/周期、固定m，原二维径向核&固定$h=1/16$，含自项；无离散归一化或核宽拟合\\
ledger实验对照&给定初始$\rho_0$、固定邻居环与有向面积比&新的最小适配器；不调用旧v3/v4，无F拟合/重锚/修补\\\bottomrule
\end{tabularx}
\par\medskip
密度及系数阶段不读取$c_0$、浓度参考/旧误差。m与$c_0$同在NPZ时只读mass键；两密度/系数包全部独立验证封存后才加载标量。
\section*{2. 核的连续归一化不等于粒子常量精确}
\[
 W_h(r)=\frac9{4\pi h^2}(1-q/2)^6(1+3q+35q^2/12),\quad q=r/h<2,\qquad
 \widehat\rho_i=\sum_jm_jW_h(r_{ij}).
\]
支撑外为零，保留自项与周期最小镜像；$2h=1/8<1/2$。输入非法直接拒绝，不抖动或隐式修补。
\[
 \int_0^2q(1-q/2)^6(1+3q+35q^2/12)\,dq=\frac29,\qquad
 2\pi\frac9{4\pi}\frac29=1.
\]
主展开积分与独立beta积分一致。均匀格上节点密度为45项有限和$\rho_*$，含自项；本轮仅给出符号表达，\textbf{未数值求值}。未来必须保留原始求和及其相对1的偏差，不能用1或$\rho_*$覆盖生成数组。
\section*{3. ledger的代理环与真实质量分开}
由旧四向邻接固定逆时针环与周期镜像；使用给定连续提升的位置，精确有理有向面积：
\[
 A_i^*(t)=\tfrac12\sum_j\operatorname{cross}(v_{ij},v_{i,j+1}),\qquad
 \rho_i^L(t)=\rho_{0,i}\frac{A_i^*(0)}{A_i^*(t)}.
\]
T1静止环是菱形，$A_i^*=2/N^2$，而$m_i=1/N^2$，故$\rho_{0,i}A_i^*=2m_i$。
这是重叠代理面积，不能替换物质质量，也不能除2制造质量一致。静止比值为1，不验证运动/压缩密度。
原历史驱动包含绝对面积、非有限回退、重锚、F拟合和弧修补；本契约不运行它们。新环退化、翻转、自交或密度非正时保存拒绝，无abs修复。
\newpage
\section*{4. 只新增必要的kernel路径}
不重建几何，用同一封存面和两半串联阻力：
\[
 g_{ij}^{s}=\frac{|S_{ij}|/r_{ij}}{1/(2\rho_{s,i}D)+1/(2\rho_{s,j}D)}.
\]
单位厚度，原D不变。密度/D按binary64的精确有理解释计算g，全部2048面独立复核。ledger必须与旧oracle的有理g逐项完全相同，才可据相同m、$c_0$、生成元和时间序列引用旧三条路径；明确标注身份复用，不声称新运行。若不同，保存差异并停止，不赋rho=1、不覆盖g或自动补ledger求解。

未来最多2密度字段、2系数包；\textbf{仅kernel新增3实际/3独立路径}，K4/8/16，共28作用区间。0新几何、0参考PDE、0旧路径重跑。两组必要软件检查先通过，才502增至504；本轮未运行，仍502。
\section*{5. 实现控制与物理偏差分列}
静态生成元具有指数半群性质，沿用原T1的时间差在原数值指标内规则，不要求接近零差下降；H差使用原$h_m(E)$。
这不修改Becker时变H下降判据，不能据此改写任何旧失败。

系数封存后才独立80/100位求$\rho_*$及$\lambda_s=\rho_*^s8DN^2\sin^2(\pi/N)$，核验实际完整图作用。符号不参与构造、不从测得衰减或g拟合、不代替实际状态。常量/特征模残差先过原$10^{-10}$代数守卫；原机器/尾项、完整独立序列差、解析评价余量及残差时间项全保留。

来源误差比较固定正项：双方K4/8、K8/16状态差，双方全部K4/8/16数值项，四倍原参考评价余量、两倍初值桥及双方残差时间项。浓度变化与$B/\sigma_{16}$比较，H误差变化与原$h_m(B)$比较；增大、减小、未分辨都报告，不预设改善、不改分母或阈值。

密度偏差、空间非均匀性、代理锚质量差、实际状态差及连续参考误差分开报告。代数$10^{-10}$不成为密度物理精度标准；符号控制通过不等于密度准确。
\section*{6. 下一项的执行边界}
先冻结适配器、驱动、两组软件检查及独立审计，再产生任何测试/密度数组。软件组检查输入语义/拒绝和同后端系数接口，各至多16节点、8帧；已知仿射夹具只作验证，不在算法中拟合F。
软件上限120秒，科学300秒/1024MiB。失败、超限、ledger身份不等或实现控制未决均保存停止，不自动增加N/K/h/模型/参考。物理偏差增大是允许报告的结果，不触发调参。

单一均匀标定不验证密度梯度、运动、压缩、接触/激波或三来源空间收敛，不能区分恒D/恒Gamma。所有Becker缺口与fluid ORC关闭状态保持。下一项已登记，\textbf{尚未执行}。
\section*{7. 核验与恢复}
主定义核验__SECONDS__秒、__RSS__MiB；独立审计__AUDITSECONDS__秒。782原文件、15附件、115旧正式清单及四组历史差异保持。当前0新物理计算/软件测试，旧502未重跑。
代码和精选结果包可重放本轮独立审计，命令见交接入口。精选包不是完整工作区，不含15附件、全部历史运行、未取回v080或聊天。仅本地PDF/ZIP和Git恢复点，未发布/推送、无新版本。
\end{document}
'''
    for key,value in {'SECONDS':f"{ex['elapsed_seconds']:.3f}",'RSS':f"{ex['peak_RSS_MiB']:.2f}",'AUDITSECONDS':f"{a['elapsed_seconds']:.3f}"}.items():tex=tex.replace('__'+key+'__',value)
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
    with zipfile.ZipFile(exports / 'LTN_becker_delta004_budget_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/t1_density_definition_protocol.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    selected.update(read(OUT/'next-gate-spec.json')['sources'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note = """LTN smooth T1 density-source contract,2026-11-07 UTC. NOT executed.
Read T1_DENSITY_SOURCE_CONTRACT.md and T1_DENSITY_SOURCE_DEFINITION.md.
Old N32 mass/c0/reference/geometry/oracle paths are sealed. No new density,coefficient,graph,state or PDE.
Future one static calibration:radial mass kernel h=1/16;new experimental fixed oriented fan-area ledger,no legacy driver/F/repair.
Continuous kernel normalization does not force discrete density1. Proxy ledger anchor mass is2m,never replace originalm.
Future2fields/2coefficient packets,3kernel actual/3independent paths,K4/8/16;ledger reuses oracle only after exact coefficient identity.
Two future software groups would raise502to504 only if both pass;current502unchanged.
Static T1 semigroup time rule does not waive unresolved Becker H. Stage3 incomplete,fluid ORC OFF.
782 originals,research/docs,selected records only;not full workspace,15 attachments,unrecoveredv080 or chat.
Verify review-manifest.json/original-handoff-sha256.json. Read-only current audit:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_t1_density_definition.py --directory runs/reconnect-t1-density-definition
Local only,no publication/push.
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
        with tempfile.TemporaryDirectory(prefix='ltn-t1-density-definition-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_t1_density_definition.py',
                '--directory', 'runs/reconnect-t1-density-definition'], cwd=directory,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 34, 'prior_formal_manifests_verified': 115,
        'attachments_verified_in_workspace': 15, 'new_graphs': 0, 'new_actual_paths': 0, 'new_independent_paths': 0, 'new_reference_PDE_solves': 0,
        'new_software_checks': 0, 'recorded_checks_total': 502, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# T1密度来源契约下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。本轮仅完成契约定义；后续单一标定尚未执行；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['document','package'])
    args=p.parse_args()
    document() if args.mode=='document' else package()
