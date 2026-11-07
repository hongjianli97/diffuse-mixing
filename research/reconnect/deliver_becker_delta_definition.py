"""Local report and verified selected recovery package for fixed-material delta design."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile
from audit_route_decision import ROOT, read, save, digest
OUT=ROOT/'runs/reconnect-becker-delta-definition'
PARENT=ROOT/'runs/reconnect-becker-common-k32'
BASE='LTN_becker_delta_definition'
DATE='20261107'
FILES=['becker_delta_definition.py','audit_becker_delta_definition.py','becker_delta_definition_protocol.json',
       'BECKER_DELTA_CONTROL_DEFINITION.md','BECKER_DELTA004_CONTRACT.md','deliver_becker_delta_definition.py']

def document():
    p=read(ROOT/'research/reconnect/becker_delta_definition_protocol.json')
    d,inv,a,ex=read(OUT/'definition.json'),read(OUT/'input-inventory.json'),read(OUT/'audit.json'),read(OUT/'execution.json')
    assert a['passed'] and ex['completed'] and not d['new_delta_data_generated']
    for name in ('main','audit'):shutil.copyfile('/tmp/ltn-delta-definition-'+name+'.log',OUT/(name+'.log'))
    sources=dict(p['sources']);seal=read(PARENT/'sha256.json')
    extra=['runs/reconnect-becker-spatial/common-observations.npz',
        'runs/reconnect-becker-spatial/n32/k8/trajectory-states.npz',
        'runs/reconnect-becker-spatial/n32/k16/trajectory-states.npz',
        'runs/reconnect-becker-common-k32/n32/k32/trajectory-states.npz',
        'runs/reconnect-becker-spatial/reference/observations-l0-k8192.npz',
        'runs/reconnect-becker-spatial/reference/observations-l1-k8192.npz',
        'runs/reconnect-becker-spatial/reference/observations-l1-k16384.npz']
    for name in extra:
        assert digest(ROOT/name)==seal[name]
        sources[name]=seal[name]
    for name in ['research/reconnect/BECKER_DELTA004_CONTRACT.md','runs/reconnect-becker-delta-definition/definition.json',
                 'runs/reconnect-becker-delta-definition/input-inventory.json','runs/reconnect-becker-delta-definition/audit.json']:
        sources[name]=digest(ROOT/name)
    save(OUT/'next-gate-spec.json',{'type':'single_delta004_fixed_material_N32_K8_16_32_diagnostic',
        **d['proposed_case'],'contract':p['contract'],'contract_sha256':digest(ROOT/p['contract']),
        'sources':dict(sorted(sources.items())),
        'problem_family':d['problem_family'],'M_star_hex':inv['fixed_M_star_hex'],
        'reuse_old_mass_labels_candidate_mass_c0_and_reference_mesh_c0':True,
        'new_reference_terminal_states_required':True,'all56_graphs_sealed_before_scalar_reference':True,
        'original_state_and_H_time_decrease_tests_required':True,
        'error_change_direction_not_preselected':True,'H_direction_agreement_is_not_magnitude_accuracy_acceptance':True,
        'no_new_delta_three_grid_or_delta_zero_claim':True,
        'stop':'record failure,time cap,time/H unresolved or response unresolved; no automatic K64,grid/reference/delta/density sweep; no clipping,normalization,F/C fit or acceptance rewrite',
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'stage3-status.json',{'date':'2026-11-07','timezone':'Etc/UTC','definition_completed':True,
        'fixed_saved_M_star':inv['fixed_M_star'],'proposed_new_delta':.04,'proposed_case_executed':False,
        'old_common_K32_spatial_diagnostic_resolved':True,'old_shifted_H_outcomes':d['old_shifted_H_outcomes'],
        'new_graphs':0,'new_candidate_paths':0,'new_reference_PDE_solves':0,'new_density_sources':0,'new_software_checks':0,
        'recorded_checks_total':502,'new_candidate_h_control_completed':False,'delta_zero_limit_claimed':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,tabularx,hyperref,fancyhdr}
\setlength{\headheight}{15pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-07 UTC}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 层厚度对照的物质定义}\\[5pt]{\large 只读设计完成；新$\delta=.04$尚未运行}\end{center}
\textbf{Stage3 incomplete，fluid ORC关闭，$\kappa$未算。}
本轮未采样新$\delta$场、未生成新图/状态、未运行参考PDE。只提出一个新厚度值，保留全部旧结果。
\section*{1. 固定质量坐标，改变层厚度}
原Becker代码有$B_\delta=b\delta$，$b=j/[(u_1+u_2)\ln9]$，$\xi_\delta(z)=\delta q(z)$。
$u_1,u_2,j$由固定上游参数决定。令$z_{0,\delta}=q^{-1}(1/\delta)$，则
\[
 z_\delta(\mu,t)=z_{0,\delta}+\frac{jt-\mu}{B_\delta},\qquad
 X_\delta(\mu,t)=u_1t-\xi_\delta(z_\delta),\qquad \rho_\delta=\frac{j}{u(z_\delta)}.
\]
由于$\xi_{\delta,z}=B_\delta u/j$，有
\[
 X_{\delta,\mu}=\frac{u}{j}=\frac1{\rho_\delta},\qquad
 X_{\delta,t}=u_1-u,\qquad
 \int_{X_\delta(\mu_i,t)}^{X_\delta(\mu_{i+1},t)}\rho_\delta\,dx=\mu_{i+1}-\mu_i.
\]
单位横截面上固定原$\mu_i$即固定物质质量。固定$M^*=\mu_{32}=__M__$，
取$c_*(\mu)=\frac12+\frac14\cos(\pi\mu/M^*)$，初始单元质量平均就是现有cos乘sinc公式。
\[
 \left.\partial_t c\right|_\mu=\partial_\mu(\rho_\delta^2D\partial_\mu c),
 \qquad \rho_\delta^2D\partial_\mu c=0\ \text{于两外端}.
\]
此问题族固定质量标签、m、质量平均c0和物理标量D；几何由新流场决定。
\section*{2. 哪些能复用，哪些不能}
\begin{tabularx}{\textwidth}{p{31mm}XX}\toprule
对象&固定或复用&必须重建/限制\\\midrule
物质状态&旧$\mu_i,m_i,c0,M^*,\sigma_{32}$及ID&不能重新用新$\mu(a)$或新flow.total造质量区间\\
候选几何&32列、8行，原质量标签中点规则&新轨迹、点密度、物质外端、共享面与阻力\\
初始物理区间&左端由锚点保持$-1$&右端不保证$+1$；物理h和Eulerian初值不能宣称不变\\
参考输入&原对齐网格、质量和c0逐位一致&三个新参考终态及其误差/观测桥必须重算\\
物理参数&上游端态、D=.02、T=.1&B、黏度与热导系数随$\delta$减半，D不缩放\\\bottomrule
\end{tabularx}

$z=\pm\ln9$对应速度10\%和90\%位置，间距
$B_\delta(u_1+u_2)\ln9/j=\delta$。流体停滞焓与标量混合指标H须区分。
\newpage
\section*{3. 两个已查明的误复用风险}
旧aligned\_becker\_check主驱动要求末端质量等于flow.total，限定原物理区间；新固定$M^*$问题族不能直接调用该驱动。
旧becker\_spatial\_core在N=32分支直接复用旧H；新$\delta$必须从新raw状态计算H。
保留旧源码，新增独立驱动并先冻结。只复用适用的底层参考、投影、普通阻力与传播函数。
\section*{4. 下一次只执行一个额外厚度}
$\delta=.04$、32列乘8行、K8/16/32：56新图，3实际/3完整独立路径。
旧$\delta=.08$相同网格/时间档复用。新图全部独立核验封存后才读取标量/参考。
三个新参考仍用旧128父区间的16486/32972/32972格和8192/8192/16384步；
原输入不变，只有密度回调与终态不同。CN monitor及原质量/极值守卫保持，不宣称无条件最大值原理。

主上限900秒/1024MiB，参考合计300秒并计入主上限。0旧重跑、0新密度重构模式，软件测试累计502。
\section*{5. 预算不调，正负结果都保留}
新误差预算记为$\mathcal B_1$，与上页Becker轮廓参数$B_\delta$区分。
\[
 \mathcal B_1=2\sum_{K=8,16,32}e_K+2R_1+b_{\rm initial}+\sum_K b_K.
\]
$e_K$保留机器、原尾项与完整独立末态差；$R_1$含新参考空间/时间差和全部初值/末态汇总及几何桥。
共同观测预算保留容量范数、初值/参考/三个时间档的汇总舍入及容量缺陷界。旧侧全部预算直接保留，不删K4等旧正项。
新时间资格仍要求状态和H差下降均严格超过原正项公式；旧时间通过不能移用于新$\delta$。

四项误差变化$E_{.04}-E_{.08}$与两侧有限时间/数值/参考指标之和比较，分类为增大、减小或未分辨。
\textbf{层变薄不预设误差降低。} 候选与参考H响应分别判别正、负或未分辨；
均可分辨且同号仅表示方向一致，幅值偏差完整报告，不等于精度通过。

新$\delta$没有候选三档h试验，空间误差不能继承旧结果或当作零。两个厚度不证明$\delta\to0$。
失败、超限、时间/H或响应未决即保存停止，不自动加K64、网格、参考、$\delta$或密度扫描；禁止裁剪、归一化、F/C拟合或改阈值。
\section*{6. 本轮核验与交付}
26份封存来源、旧质量/初值与嵌套参考输入核对，独立标准库解析NPZ并检查坐标链式求导、层厚代数系数和物质PDE量纲。
只读主流程__SEC__秒/__RSS__MiB；未生成任何新$\delta$数值。782原文件、15附件、112旧正式清单与历史修正保持。

PDF、代码/精选结果ZIP及Git恢复点仅本地保存，不发布/推送。ZIP不是全工作区，不含15上传附件、未取回v080或完整聊天。
\begin{verbatim}
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/audit_becker_delta_definition.py \
--directory runs/reconnect-becker-delta-definition
\end{verbatim}
\end{document}
'''
    tex=tex.replace('__M__',format(inv['fixed_M_star'],'.16g')).replace('__SEC__',f"{ex['elapsed_seconds']:.3f}").replace('__RSS__',f"{ex['peak_RSS_MiB']:.2f}")
    with (OUT/f'{BASE}_report_{DATE}.tex').open('x') as f:f.write(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n113. [固定物质质量坐标的层厚度对照定义](../research/reconnect/BECKER_DELTA_CONTROL_DEFINITION.md)（2026-11-07 UTC）。\n\n'
        '只读定义完成，未采样或计算新δ。固定旧μ标签/M*/m/c0，重建新δ轨迹、密度、物质边界与参考终态；不能同时固定原Eulerian区间/初场。明确旧flow.total断言和N32旧H复用分支不能用于新δ驱动。'
        '冻结下一项唯一δ=.04、32列×8行、K8/16/32，最多56图、3实际/3独立与3新参考；旧参考网格/c0复用，主900秒/1024MiB、参考300秒。原状态/H时间资格与全部正项不改，误差增减均报告，不预设变薄改善，不宣称新δ三档h或δ→0。'
        '26封存来源、旧输入及代数系数/量纲独立核验；782原文件、15附件、112旧正式清单和历史修正保持。当前0新图/路径/参考/密度/软件检查，累计502。下一项尚未执行；Stage3 incomplete、fluid ORC关闭，仅本地PDF/精选ZIP与Git恢复点，不发布/推送。\n')


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
    with zipfile.ZipFile(exports / 'LTN_becker_common_k32_code_results_20261107.zip') as z:
        selected = set(json.loads(z.read('review-manifest.json')))
    for folder in ('research', 'docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    selected.update(read(ROOT / 'research/reconnect/becker_delta_definition_protocol.json')['sources'])
    selected.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    selected.update([str(pdf.relative_to(ROOT)), str((PARENT / 'sha256.json').relative_to(ROOT)), 'GITHUB_HANDOFF.md'])
    originals = read('/workspace/original-handoff-sha256.json')
    selected.update(originals)
    manifest = {p: digest(ROOT / p) for p in sorted(selected)}
    for p, h in originals.items():
        assert manifest[p] == h
    note = """LTN fixed-material delta-control definition,2026-11-07 UTC.
Read-only26 sealed sources and original material/reference arrays. No delta=.04 sampling,graphs,states or PDE solves.
One proposed delta=.04 case:fixed old mass labels,Mstar,m,c0;32columns,8rows,K8/16/32;56 new graphs,3actual/3independent and3new reference terminals,NOT executed.
Old reference meshes/c0 reusable;old reference terminals/H cannot be relabelled as new. Material geometry/domain changes must be reported.
Original H and positive budgets retained;no clipping,renormalization,F/C fit. No new-delta space-convergence or delta->0 claim.
782 originals,research/docs,selected records only;not full workspace,15 attachments,unrecoveredv080 or chat.
Verify review-manifest.json/original-handoff-sha256.json. Replay read-only audit:
PYTHONDONTWRITEBYTECODE=1 python research/reconnect/audit_becker_delta_definition.py --directory runs/reconnect-becker-delta-definition
Stage3 incomplete,fluid ORC OFF;recorded tests502. Local only,no publication/push.
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
        with tempfile.TemporaryDirectory(prefix='ltn-delta-definition-zip-', dir='/tmp') as directory:
            z.extractall(directory)
            result = subprocess.run([sys.executable, 'research/reconnect/audit_becker_delta_definition.py',
                '--directory', 'runs/reconnect-becker-delta-definition'], cwd=directory,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            for p, h in manifest.items():
                assert digest(Path(directory) / p) == h, p
    receipt = {'date': '2026-11-07', 'timezone': 'Etc/UTC',
        'PDF': {'path': str(pdf.relative_to(ROOT)), 'pages': 2, 'bytes': pdf.stat().st_size, 'sha256': digest(pdf)},
        'ZIP': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'included_files': len(manifest), 'CRC_SHA_and782_originals_verified': True,
        'fresh_replay_exit_code': 0, 'fresh_replay_log': result.stdout, 'extracted_files_unchanged': True,
        'source_files_verified': 26, 'prior_formal_manifests_verified': 112,
        'attachments_verified_in_workspace': 15, 'new_graphs_or_states_or_reference_solves': 0,
        'new_software_checks': 0, 'recorded_checks_total': 502, 'next_gate_executed': False,
        'stage3': 'incomplete', 'fluid_orc_enabled': False, 'published_or_pushed': False}
    save(exports / f'{BASE}_delivery_{DATE}.json', receipt)
    with (exports / f'{BASE}_downloads_{DATE}.md').open('x') as f:
        f.write(f'# 层厚度对照定义下载\n\n[两页PDF]({pdf.name}) · [代码与精选结果ZIP]({archive.name})\n\n'
                f'PDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n'
                'Stage3 incomplete，fluid ORC关闭。下一项δ=.04单例尚未执行；仅本地保存，未发布/推送。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fresh_replay_log'}, indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['document','package'])
    args=p.parse_args()
    document() if args.mode=='document' else package()
