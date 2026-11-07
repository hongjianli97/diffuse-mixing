"""Local prototype report, frozen next gate, and independently replayed recovery ZIP."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-aligned-reference';SOURCE=ROOT/'runs/reconnect-initial-projection'
BASE='LTN_aligned_reference';DATE='20261106'
FILES=['aligned_mass_reference.py','test_aligned_reference.py','aligned_reference_study.py','aligned_reference_protocol.json',
       'ALIGNED_REFERENCE_PROTOTYPE_CONTRACT.md','audit_aligned_reference.py','deliver_aligned_reference.py','ALIGNED_REFERENCE_PROTOTYPE.md']


def document():
    r=read(OUT/'results.json');a=read(OUT/'audit.json');assert a['passed'] and r['software_checks_passed']==2
    for label in ('main','audit'):shutil.copyfile(f'/tmp/ltn-aligned-reference-{label}.log',OUT/f'{label}.log')
    old=read(ROOT/'research/reconnect/becker_reference_protocol.json')
    sources=old['sources']+[{'path':p,'sha256':digest(ROOT/p)} for p in (
      'research/reconnect/aligned_mass_reference.py','research/reconnect/ALIGNED_REFERENCE_PROTOTYPE_CONTRACT.md',
      'runs/reconnect-aligned-reference/results.json','runs/reconnect-aligned-reference/audit.json',
      'runs/reconnect-initial-projection/results.json','research/reconnect/t1_uniform_core.py')]
    save(OUT/'next-gate-spec.json',{'already_executed':False,'type':'one_fixed_material_aligned_Becker_reference_only_L_shape',
      'sources':sources,'physical_case':old['physical_case'],
      'mesh':'original32 material mass intervals unchanged; h=exact total mass/16384; n_i=ceil(exact m_i/h); nested fine counts=2*n_i. Rational construction from saved endpoints, no inserted slivers or normalization. Reject nonrepresentable cells',
      'reference_levels':[{'child_count_level':0,'steps':8192},{'child_count_level':1,'steps':8192},{'child_count_level':1,'steps':16384}],
      'base_cell_count_range':[16384,16415],'fine_cell_count_range':[32768,32830],
      'reference_method':'frozen isolated aligned_mass_reference.py, midpoint CN, rho^2 D at mass face/centre distance, zero exterior scalar flux; unchanged Becker density and analytic finite-interval initial means',
      'positivity_policy_fixed_before_physical_inputs':'monitor; record every failure of sufficient RHS-diagonal condition, keep original finite/mass/extrema guards. This is monitored reference integration, not unconditional maximum-principle certification. Never switch after failure',
      'reference_guards':old['reference_guards'],
      'observation':'sum complete child mass integrals over each original parent, divided by original parent mass; independent Fraction sums; retain all production-vs-exact aggregation and child mass geometric closure differences positively',
      'reference_indicator':'norm_m(base8192-fine8192)+norm_m(fine8192-fine16384), original32 column mass; empirical only, no Richardson or certified PDE-error claim',
      'fixed_positive_budget':'retain every original candidate numeric/observation component; in this separate result replace original reference component by 2*(new space+time). Add 2*sum(all3 initial-mean bridges), 2*sum(all3 terminal production/exact aggregation bridges), and 2*sum(all3 initial+terminal child geometric-mass defect bounds). No cancellation or deletion; old result budgets unchanged',
      'geometry_bridge':'for each parent use sum(abs(binary64 child mass - exact endpoint difference)*abs(saved child mean))/original parent mass, then original mass norm; evaluate both initial and terminal. Retain nonzero mismatch, do not reweight',
      'comparison':'saved original32x8 candidate K4/8/16 only; unchanged original sigma,H helper and strict greater-than rule. Retain saved H and state differences, report any floating re-evaluation differences separately; old two unresolved decisions remain',
      'new_reference_PDE_solves':3,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'old_candidate_reruns':0,
      'new_software_checks':0,'recorded_checks_total':502,
      'no_true_internal_reference_faces_to_candidate_builder':True,'main_wall_seconds_cap':300,'main_peak_RSS_MiB_cap':1024,
      'resource_basis':'previous uniform three reference solves about62s; aligned overhead not yet measured. Caps are limits, not performance claims',
      'audit_scope':'independent exact complete-child mass/observation and 80/100-digit positive budget audit of saved endpoints/diagnostics; no fourth PDE, no independent all-time-step validation claim',
      'stop':'preserve failure/unresolved and stop; no fourth reference, automatic N/K/parameter scan, clipping, normalization, F/C fitting or altered H/threshold. A resolved temporal comparison still does not close candidate spatial convergence or all Stage3',
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-06','timezone':'Asia/Shanghai','decision':r['decision'],'recorded_checks_total':502,
      'new_software_checks':2,'new_Becker_reference_solves':0,'new_candidate_transient_runs':0,'next_physical_reference_executed':False,
      'old_H_indicator':9.468091355897298e-7,'old_H_decline':5.762795999286752e-7,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    report='''# 完整物质子单元参考原型：两项软件验证通过

2026-11-06（Asia/Shanghai）。按[上一轮登记](../../runs/reconnect-initial-projection/next-gate-spec.json)实现隔离的非均匀质量参考原型。[实现前契约](ALIGNED_REFERENCE_PROTOTYPE_CONTRACT.md)与两个夹具的输入在求解前冻结。

**本轮完成软件原型，未执行新Becker参考或候选传播。** 两项预定检查首次通过，记录从500增至502，旧500项未重跑。主过程19.764秒、54.46MiB（含历史完整性核验）。合成数据共7次求解器尝试、33次浮点线性求解，含2次预期拒绝；不是7个新物理算例。

## 数学和实现

固定物质质量坐标区间由完整子区间组成，原父边界和质量保留；后继层子数翻倍。精确有理数划分后一次转binary64，不可表示则拒绝。未知量是每个子区间质量平均，父观测为Σm_j c_j/原m_i。无部分格插值、质量或状态重归一化，常数和质量浮点差保留。

独立参考方程为 M dc/dt=−L(t)c，共享g=(ρ²D)_face/相邻质量中心距，外面零通量。此为新细参考原型，不取代候选普通中心阻力后端或旧均匀参考代码，不给候选提供真实内部物质面。

面左右宽度a,b上的真实均值满足

(c̄_R−c̄_L)/[(a+b)/2]=c′+(b−a)c″/3+(a²−ab+b²)c‴/12+O(h³)。

固定宽度比突变一般有一阶局部通量误差；不能直接宣称任意非均匀网格二阶。固定父质量、每父区间n_i=ceil(m_i/h)等分时，格宽差O(h²/min m_i)，光滑条件下仅给出条件性二阶面通量一致性，不证明全局解阶。本轮未生成生产网格或测新物理空间阶。

CN满足(M+dt L/2)c⁺=(M−dt L/2)c⁻。右矩阵对角m_j−dtΣg/2≥0是保持非负的充分条件。默认require不满足即求解前拒绝；显式monitor记录条件失败，每步仍检查原始有限性、质量漂移≤1e-9和初值极值±1e-10，失败保留原状态并拒绝。monitor不是最大值原理证明，不得见到失败后换策略。D=0直接精确恒等，不求解或修补。

## 固定夹具和独立保存数据审计

| 检查 | 结果 |
|---|---|
| 单位质量域16格/8步/常密度，新旧参考端点 | 逐位一致 |
| 非均匀父区间[0,1/8,3/8,1]，8/16个嵌套子格 | 原边界、完整子质量积分及嵌套关系通过 |
| 两层解析初值汇总与原父均值的质量范数差 | 0；5.55111512e-17 |
| 常数最大浮点差；D0 | 2.77555756e-16；逐位恒等 |
| 二次真均值梯度非均匀首项公式最大差 | 2.22044605e-16 |
| 3格/大步长require | 0次线性求解，预先拒绝 |
| 同一反例monitor | 原始最小值−0.7345464834，保留后拒绝 |

合成舍入比较的64eps尺度在执行前已固定，不用于原H物理判据。独立审计使用Fraction Thomas消元重建24个已保存正常步和1个拒绝步的方程，输入是各步保存的前态和系数，不再次时间积分。正常步最大局部舍入差2.338223e-16，拒绝步9.213415e-16；每个精确方程质量守恒。独立核对解析系数、完整子积分、嵌套观测和80/100位初始均值。这里验证的是保存方程和观测，不认证终态PDE误差。

## 旧结果与下一步

原两次未决保持：H指标9.46809136e-7仍高于下降5.76279600e-7；未改H、阈值、任何旧正项或旧状态。782原文件、15附件、105份旧正式清单保持，历史失败/恢复保留。Stage3 incomplete，fluid ORC关闭，κ未计算；三密度来源也尚未闭合。

下一项单独冻结为**物质对齐Becker参考L形三点**，本轮尚未执行：δ=.08、D=.02、T=.1，复用原32列及K4/8/16候选。h=M/16384，各父区间子数ceil(m_i/h)，下一层翻倍；时间组8192/8192/16384。预先选择monitor并保留原守卫，记录充分条件是否满足。按完整子积分比较，所有初值、观测、质量几何浮点差继续作为正项；旧未决不追认。仅3个参考解，不新建或重跑候选，不扩扫描；细化差仍非认证误差界。详见[next-gate-spec.json](../../runs/reconnect-aligned-reference/next-gate-spec.json)。即使新时间判别通过，也不等于移动T4或Stage3完整通过。

## 本地交付和恢复

两页PDF、研究代码/文档、782原文件及精选父例/本轮结果ZIP；不是39GB完整镜像，不含15上传附件、未取回v080或聊天。仅本地保存与提交，不发布/推送。ZIP的CRC/SHA和全新解压保存数据复核见交付JSON。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/audit_aligned_reference.py \\
  --directory runs/reconnect-aligned-reference
```
'''
    with (ROOT/'research/reconnect/ALIGNED_REFERENCE_PROTOTYPE.md').open('x') as f:f.write(report)
    tex=r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-06}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 完整物质子单元参考原型}\\[4pt]{\large 两项软件验证通过，尚未运行新物理参考}\\[3pt]2026年11月6日（Asia/Shanghai）\end{center}
\section*{1. 本轮完成内容}
实现隔离的非均匀质量参考求解器，保留原父物质边界，将每个父区间分为完整子区间。
下一层各子数翻倍；精确有理数划分后一次转换为binary64，不可表示则拒绝。
未知量是物质质量平均，父观测为
\[\bar C_i=\frac{\sum_{J_j\subset K_i}m_j C_j}{m_i}.\]
无部分细格插值或重归一化，所有浮点差保留。新原型只供独立细参考，旧候选和旧均匀参考源码不变。
两项预定软件检查首次通过，累计502，旧500未重跑。本轮\textbf{0个新Becker参考、0个候选图/传播}。
\section*{2. 非均匀通量的精度限制}
参考方程为$M\dot c=-Lc$，共享$g=(\rho^2D)_{\rm face}/\Delta\mu_{\rm centre}$，外面零通量。
左右宽度$a,b$上的真实单元均值满足
\[\frac{\bar c_R-\bar c_L}{(a+b)/2}=c'_f+\frac{b-a}{3}c''_f+
\frac{a^2-ab+b^2}{12}c'''_f+O(h^3).\]
固定宽度比突变一般只有一阶局部通量一致性，不能宣称任意非均匀网格二阶。
固定父质量下若$n_i=\lceil m_i/h\rceil$，父内等分可使宽度差$O(h^2/\min m_i)$；
这只给出光滑条件下的面通量结论，不证明全局解阶。本轮没有生产网格或物理收敛结果。
\section*{3. 时间格式必须有明确限制}
\[(M+\Delta t L/2)c^+=(M-\Delta t L/2)c^-.\]
右矩阵对角$m_j-\Delta t\sum_k g_{jk}/2\ge0$是CN保持非负的充分条件。
默认require在不满足时求解前拒绝；显式monitor记录条件失败，并检查每步原始有限性、
质量漂移$\le10^{-9}$及初值极值$\pm10^{-10}$。越界保留原状态后拒绝，不裁剪。
monitor\textbf{不是无条件最大值原理证明}，不得见到失败后换策略。$D=0$直接恒等，无求解或修补。
\newpage
\section*{4. 固定小型夹具及独立审计}
\begin{center}\begin{tabular}{ll}\toprule
检查&结果\\\midrule
均匀16格/8步，新旧终态&逐位一致\\
非均匀8/16子格、完整与分层汇总&通过\\
两层初始观测桥质量范数&0；$5.55111512\times10^{-17}$\\
常数最大浮点差&$2.77555756\times10^{-16}$\\
零扩散&逐位恒等\\
大步长require&求解前拒绝\\
大步长monitor&原始最小值$-0.7345464834$，拒绝\\\bottomrule
\end{tabular}\end{center}
主过程19.764秒/54.46MiB，含历史完整性核验。共7次合成求解器尝试、33次浮点线性求解，含2次预期拒绝。
独立审计用精确有理数重建25个保存线性方程，没有重新时间积分。
正常步最大局部舍入差$2.338223\times10^{-16}$，拒绝步$9.213415\times10^{-16}$，
精确方程质量守恒；解析系数、完整子积分及80/100位初始均值核验通过。
软件舍入比较尺度预先固定为$64\epsilon$，不用于原物理H判据，不认证终态PDE误差。
\section*{5. 旧未决保持；下一项已登记但未执行}
旧H指标$9.46809136\times10^{-7}$仍高于下降$5.76279600\times10^{-7}$。
782原文件、15附件、105旧正式清单及历史失败恢复记录保持。
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。}

下一项仅3个物质对齐Becker参考：$\delta=.08,D=.02,T=.1$，原32列不变；
$h=M/16384$，每父区间$n_i=\lceil m_i/h\rceil$，细层子数翻倍，时间步数组合8192/8192/16384。
预先选monitor并保留原守卫和充分条件诊断，完整子积分观察；初值、观测和几何质量差均作正项。
只读取原K4/8/16候选，H公式/阈值不改，失败或未决即停，不扩扫描。
即使新时间判别通过，也不证明候选空间收敛、移动T4及三密度来源全部闭合。
\section*{6. 本地交付与保存数据复核}
ZIP含代码/文档、782原文件、精选父例及本轮结果；不是39GB全镜像，不含15附件、未取回v080或聊天。
本轮未发布/推送。恢复后的保存数据核验入口：
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_aligned_reference.py \
--directory runs/reconnect-aligned-reference
\end{verbatim}
\end{document}
"""
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n106. [完整物质子单元参考原型](../research/reconnect/ALIGNED_REFERENCE_PROTOTYPE.md)（2026-11-06）。\n\n'
      '隔离非均匀质量参考实现完成，两项预定小型软件检查首次通过，累计502，旧500未重跑。均匀极限与旧参考逐位一致；完整子积分/嵌套/常数/D0及CN两种拒绝策略通过。7次合成调用、33次浮点线性求解，含2次预期拒绝；monitor大步长原始最小值−0.734546并保留。独立有理数重建25个保存方程及80/100位初值审计首次通过，没有再次时间积分。任意非均匀二阶或CN无条件极值保持均不声称。0新Becker参考/候选，旧H未决与预算保持；782原文件、15附件、105旧正式清单保持。下一项仅登记对齐参考L形三点及原正项/H核验，尚未执行；Stage3 incomplete、fluid ORC关闭。仅本地PDF/代码ZIP交付与Git备份，不发布/推送。\n')


def package():
    pdf=ROOT/'exports'/f'{BASE}_report_{DATE}.pdf';archive=ROOT/'exports'/f'{BASE}_code_results_{DATE}.zip';seal=OUT/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal))
    assert read(OUT/'execution.json')['first_attempt_passed'] and read(OUT/'audit.json')['passed']
    info=subprocess.run(['pdfinfo',str(OUT/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           2' in info
    log=(OUT/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(OUT/pdf.name,pdf);save(OUT/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_visually_reviewed':True,'no_overfull_or_missing_characters':True})
    parent=SOURCE/'sha256.json';entries=read(parent)
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    entries[str(parent.relative_to(ROOT))]=digest(parent)
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in OUT.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(ROOT/'exports/LTN_initial_projection_code_results_20261106.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file());selected.update([str(pdf.relative_to(ROOT)),str(parent.relative_to(ROOT))])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN material-aligned reference prototype,2026-11-06.
Two registered synthetic software fixtures passed; recorded count502,old500 not rerun.7 solver attempts,33 floating linear solves,2 expected rejections.
Independent exact-rational audit reconstructs25 saved equations without new time integrations. Uniform legacy bitwise equivalence,complete-child observation and explicit CN positivity limitations checked.
0 new Becker reference or candidate solves. Old unresolved results/H preserved;Stage3 incomplete,fluid ORC OFF.
Next gate is a separate fixed three-reference aligned Becker check,NOT yet executed. No automatic scans.
Contains research/docs,782 originals,selected parent results and current prototype. Not full39GB workspace,15 attachments,missingv080 or chat.
Only local delivery,no publishing/pushing. ZIP uses review-manifest.json;full Git backup metadata is not assumed included.
Replay saved-data audit:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_aligned_reference.py --directory runs/reconnect-aligned-reference
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-aligned-reference-restore-',dir='/tmp') as directory:
            z.extractall(directory);replay=subprocess.run([sys.executable,'research/reconnect/audit_aligned_reference.py','--directory','runs/reconnect-aligned-reference'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    delivery={'date':'2026-11-06','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
      'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
      'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':replay.stdout,'extracted_files_unchanged':True,
      'main_and_audit_first_attempt_passed':True,'new_software_checks':2,'recorded_checks_total':502,'synthetic_solver_attempts':7,
      'synthetic_floating_linear_solves':33,'expected_rejections':2,'new_Becker_reference_solves':0,'new_candidate_transient_runs':0,
      'old_decisions_and_budgets_preserved':True,'next_physical_reference_executed':False,'stage3':'incomplete','fluid_orc_enabled':False,
      'prior_formal_manifests_verified':105,'attachments_verified_in_workspace':15,'published_or_pushed':False}
    save(ROOT/'exports'/f'{BASE}_delivery_{DATE}.json',delivery)
    with (ROOT/'exports'/f'{BASE}_downloads_{DATE}.md').open('x') as f:f.write(f'# 物质对齐参考原型下载\n\n[两页PDF]({pdf.name}) · [代码与结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n仅软件原型验证，Stage3未完成、fluid ORC关闭；未发布/推送。\n')
    print(json.dumps({k:v for k,v in delivery.items() if k!='fresh_replay_log'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['document','package']);args=parser.parse_args()
    document() if args.mode=='document' else package()
