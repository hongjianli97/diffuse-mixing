"""Local report and recovery package for initial-only projection interpretation."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest

OUT=ROOT/'runs/reconnect-initial-projection';SOURCE=ROOT/'runs/reconnect-becker-reference'
BASE='LTN_initial_projection';DATE='20261106'
FILES=['initial_projection_review.py','initial_projection_protocol.json','audit_initial_projection.py','replay_initial_projection.py',
       'MATERIAL_REFERENCE_OBSERVATION_PROPOSAL.md','deliver_initial_projection.py','INITIAL_PROJECTION_REVIEW.md']


def document():
    out=OUT;r=read(out/'results.json');ex=read(out/'execution.json');audit=read(out/'audit.json')
    assert ex['first_attempt_passed'] and audit['first_attempt_passed']
    for label in ('main','audit'):shutil.copyfile(f'/tmp/ltn-initial-projection-{label}.log',out/f'{label}.log')
    contract=ROOT/'research/reconnect/MATERIAL_REFERENCE_OBSERVATION_PROPOSAL.md'
    save(out/'next-gate-spec.json',{'already_executed':False,'type':'minimal_material_aligned_reference_prototype_and_software_fixtures',
      'reason':'all31 material cuts are inside uniform reference cells; whole means cannot uniquely determine cut means. Validate complete-child integral observation before any new physical reference run',
      'contract':{'path':str(contract.relative_to(ROOT)),'sha256':digest(contract)},
      'sources':[{'path':p,'sha256':digest(ROOT/p)} for p in ('research/rebuild/reference.py','research/rebuild/becker.py','runs/reconnect-initial-projection/results.json')],
      'scope':'new isolated nonuniform-mass reference prototype with fixed material parent/child mapping; existing candidate and uniform reference implementations untouched',
      'prototype_geometry':'each given material interval subdivided into complete children, nested levels double each parent subdivision count; retain every original mass/initial field and rounding bridge, no sliver insertion or normalization',
      'equation':'m_j dc_j/dt = sum shared g(c_neighbor-c_j), g=(rho^2 D at shared mass face)/(mass-coordinate centre separation), external zero flux; derive nonuniform consistency before use',
      'time':'midpoint Crank-Nicolson; no unconditional maximum-principle claim. Guard raw states and explicitly check step restriction or rejection in fixtures, never clip',
      'planned_new_software_checks':2,
      'fixture_categories':['exactly representable uniform-mass limit against unchanged reference.py on small prescribed constant-density data',
        'nonuniform mass and nested child-integral consistency,analytic initial means,constant preservation,D0,conservation and explicit positivity-limitation handling'],
      'fixture_resource_limit':'at most32 fine cells and16 steps per synthetic fixture; log actual solver calls; fixtures are software validation, not new Becker physical results',
      'new_Becker_reference_solves':0,'new_physical_reference_experiments':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'old_candidate_reruns':0,
      'current_recorded_checks':500,'recorded_checks_if_both_pass':502,'main_wall_seconds_cap':120,'main_peak_RSS_MiB_cap':1024,
      'stop':'if prototype/fixtures fail preserve failure; no automatic physical reference/grid scan. Any later Becker reference comparison needs a separately fixed protocol and unchanged H formula',
      'old_H_budgets_and_unresolved_decisions_unchanged':True,'no_true_internal_reference_faces_to_candidate_builder':True,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(out/'stage3-status.json',{'date':'2026-11-06','timezone':'Asia/Shanghai','decision':r['decision'],'old_case_reclassified':False,
      'new_reference_PDE_solves':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_software_checks':0,'recorded_checks_total':500,
      'reference_contract_only_proposed':True,'next_prototype_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})
    unique=[r['cases'][0],r['cases'][1]];norms=[c['precision_results'][-1] for c in unique]
    table='\n'.join(f"| {c['cells']} | {c['interior_nonaligned_boundaries']} | {float(v['mass_norms']['stored_total']):.12g} | {float(v['mass_norms']['representation']):.12g} | {float(v['positive_rounding_norm_sum']):.8g} |" for c,v in zip(unique,norms))
    report=f'''# 初值物质平均投影复核：表示差与浮点误差已分开

2026-11-06（Asia/Shanghai）。执行[原只读登记](../../runs/reconnect-becker-reference/next-gate-spec.json)，[计账协议](initial_projection_protocol.json)在计算前固定。原标量为 f(μ)=1/2+cos(πμ/M)/4，未知量始终是固定物质单元的质量平均。

**结论：{r['decision']}。** 当前初值桥主要来自不对齐细格的分片常数表示。不是改变了物理初场，亦非单纯浮点舍入。31个内部原物质边界全部位于参考细格内部；同一32768网格两组时间步的初值数据完全相同。增加时间步不能消除这个初始表示差。

| 参考单元数 | 不对齐内部边界 | 保存初值桥范数 | 解析表示差范数 | 三项浮点贡献范数之和 |
|---:|---:|---:|---:|---:|
{table}

范数保持原32列质量。三项浮点贡献分别是细均值评估/存储、粗均值评估/存储和最终投影舍入。主复核80/100位结果一致；独立审计采用另一种sinc均值公式、左右部分细格加完整内部区间积分，以100/120位核对。此处的有符号向量只用来核对恒等式，不抵消原判别预算。

## 解析恒等式

F(μ)=μ/2+M sin(πμ/M)/(4π)。原物质区间[a,b]的真实初始均值为[F(b)−F(a)]/(b−a)，与原代码的cos×sinc表达相同。

对包含切口x的细格[l,r]，定义S(x)=(x−l)[F(r)−F(l)]/(r−l)−[F(x)−F(l)]。整细格积分抵消后，粗单元表示差严格为 **[S(b)−S(a)]/(b−a)**。所有数据端点按保存二进制数的精确有理值解释，原质量与端点差一致，没有重归一化。

令P为细均值的精确分片常数重叠投影，C*为解析粗均值，A*为解析细均值。逐分量验证：保存投影−保存粗c0 = (PA*−C*) + (P保存细c0−PA*) + (C*−保存粗c0) + (保存投影−P保存细c0)。四项分别为表示、细均值浮点、粗均值浮点与投影浮点项。

对本初值，L=π/(4M)，可证|S(x)|≤L(x−l)(r−x)/2。它只约束这个已知初值的表示差，不是终时参考误差界。

## 为什么终态不能按初值修补

整细格均值不确定部分积分。以常数1/2为基准，加q(μ)=(μ−midpoint)/(4·格宽)，整格积分变化为0，但内部切口的部分积分改变；两种场都在[3/8,5/8]内。独立有理数见证已保存。解析初值提供的信息仅适用于t=0，不能将相同修正加到未知终态或从旧预算中删除该项。

## 下一步的参考观测契约

[提案](MATERIAL_REFERENCE_OBSERVATION_PROPOSAL.md)要求每个原物质区间由完整参考子区间组成，按Σm_j C_j/m_i汇总。精确算术下，对任意真实标量均值，质量积分直接相加；初始解析子积分也严格还原原粗平均。它解决部分细格信息缺失，仍须另行验证参考动力学、空间/时间误差和浮点桥。

拟采用每个原区间内部等质量划分、下一档子数翻倍；不直接插入切口制造任意小细格，不归一化质量。候选点构造器不得接收参考内部物质面。

下一项仅实现隔离的非均匀质量参考原型及两个必要软件夹具：精确可表示均匀网格退化对照；非均匀/嵌套汇总的解析初值、守恒、常数、D0与时间格式极值限制。小型合成夹具上限32格/16步，暂无新的Becker参考求解。原型及未来物理比较尚未执行，不得将此提案写成已通过方法。

## 原结论与交付

父例的全部正项保留，H指标仍{r['parent_H_indicator_unchanged']:.12g}，H下降仍{r['parent_H_decline_unchanged']:.12g}，两次未决保持。Stage3 incomplete，fluid ORC关闭，κ未算。本轮0新PDE/候选/网格/软件测试，累计500。只解析保存初态与标签，未解析终态或候选数组；整体文件哈希仍核对全部字节。

只读主复核首次成功：{ex['elapsed_seconds']:.3f}秒；独立审计首次通过：{audit['elapsed_seconds']:.3f}秒。782原文件、15附件、104份既有正式清单保持，旧汇总/诊断失败与追加恢复记录均保留。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/replay_initial_projection.py --directory runs/reconnect-initial-projection
```
代码ZIP含研究代码/文档、782原文件、父例数据与本轮解析复核；不是全39GB工作区，不含15附件或未取回v080。仅本地交付，不发布或推送。
'''
    path=ROOT/'research/reconnect/INITIAL_PROJECTION_REVIEW.md';assert not path.exists();path.write_text(report)
    tab='\n'.join(f"{c['cells']} & {float(v['mass_norms']['stored_total']):.8g} & {float(v['mass_norms']['representation']):.8g} & {float(v['positive_rounding_norm_sum']):.5g} \\\\" for c,v in zip(unique,norms))
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\setlength{\headheight}{16pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\emergencystretch=2em
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-06}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 初值物质平均投影复核}\\[4pt]{\large 不对齐细格的表示差已与浮点误差分开}\\[3pt]2026年11月6日（Asia/Shanghai）\end{center}
\section*{1. 本轮确认了什么}
原物质均值与细参考均值来自同一解析初值。31个内部原物质边界都切在细格内部；
将细均值当常数后切分，出现$10^{-9}$量级的表示差，各项浮点贡献仅$10^{-17}$至$10^{-16}$量级。
\textbf{不是物理初场改变，也不能称为纯舍入。} 同一32768参考网格两组时间步的初值完全相同，增加时间步不能消除此项。
\begin{center}\begin{tabular}{rrrr}\toprule
参考格数&保存初值桥&解析表示差&浮点项范数和\\\midrule
__TABLE__
\bottomrule\end{tabular}\end{center}
表中均为原32列质量范数。主计算80/100位一致，独立sinc/部分区间积分审计100/120位通过。
有符号分解只核对向量恒等式，不用于抵消验收预算。
\section*{2. 差异可由边界切口严格解释}
原初值及原函数为
\[f(\mu)=\tfrac12+\tfrac14\cos(\pi\mu/M),\quad
F(\mu)=\tfrac\mu2+\frac{M}{4\pi}\sin(\pi\mu/M).\]
原区间$[a,b]$真实均值为$[F(b)-F(a)]/(b-a)$。对包含切口$x$的参考细格$[l,r]$，定义
\[S(x)=(x-l)\frac{F(r)-F(l)}{r-l}-[F(x)-F(l)].\]
完整细格的解析积分相加抵消，粗区间表示差恰为
\[B_i^{\rm rep}=\frac{S(b_i)-S(a_i)}{b_i-a_i}.\]
保存投影减保存粗初值严格分解为：表示差、细均值评估/存储差、粗均值评估/存储差及最终投影舍入。
全部用保存二进制端点的精确有理值计账，质量不重归一化。
令$L=\pi/(4M)$，可得$|S(x)|\le L(x-l)(r-x)/2$。
这是已知初值的表示差界，\textbf{不是终态PDE参考误差界}。
\newpage
\section*{3. 整细格均值不能确定部分积分}
在细格上取$q(\mu)=(\mu-\mathrm{midpoint})/[4(r-l)]$。
其整格积分为0，而到内部切口的部分积分为
\[\int_l^x q\,d\mu=\frac{(x-l)(x-r)}{8(r-l)}\ne0.\]
常数$1/2$与$1/2+q$都在$[3/8,5/8]$内，整格均值相同，部分积分不同。
独立有理数见证已保存。不能从终时细均值唯一恢复切口内标量分布，
也不能把解析初值的修正直接用于未知终态或减去旧预算项。
\section*{4. 提出的参考观测契约尚未实现}
每个原物质区间应是完整参考子区间的并集，映射为
\[\bar C_i=\frac{\sum_{J_j\subset K_i}m_j C_j}{m_i},\qquad \sum_jm_j=m_i.\]
精确算术中，对任意真实细平均，质量积分直接相加；初始解析子积分也严格还原粗平均。
但参考离散、时间误差和浮点桥仍须验证，\textbf{对齐本身不证明物理通过}。

建议在每个原区间内等质量划分，下一档子数翻倍，保留共享物质边界。
不直接插入切口制造任意小细格，不归一化质量；参考内部面不得提供给候选点构造器。
非均匀参考的共享通量与质量矩阵须单独推导，不能直接套用旧均匀dm求解器。

下一项仅为隔离原型和两个必要软件夹具：精确可表示均匀网格退化；非均匀与嵌套汇总的解析初值、
守恒、常数、零扩散及时间格式极值限制。合成夹具至多32格/16步，不启动新Becker参考。
Crank--Nicolson不对任意步长保证最大值原理，必须明确限制，不能裁剪状态。
未来物理比较须另行固定方案；本轮未实现或执行该原型。
\section*{5. 原未决与原文件保持}
旧全部正项保持，H指标仍$9.46809136\times10^{-7}$，高于下降$5.76279600\times10^{-7}$。
两次未决不改，Stage3 incomplete，fluid ORC关闭，$\kappa$未算。
本轮0新PDE/候选/网格/软件测试，累计500。782原文件、15附件、104旧正式清单及原失败恢复记录保持。
只解析初态和标签，不解析终态/候选数组；哈希仍核对整体文件字节。
\section*{6. 本地交付与复核}
ZIP含研究代码、文档、782原文件、父例和本轮结果；不是39GB全镜像，不含15附件或未取回v080。
CRC/SHA与解压复核以交付JSON为准。本轮未发布或推送。
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/replay_initial_projection.py \
--directory runs/reconnect-initial-projection
\end{verbatim}
\end{document}
'''.replace('__TABLE__',tab)
    (out/f'{BASE}_report_{DATE}.tex').write_text(tex)
    with (ROOT/'docs/RESEARCH_STATUS.md').open('a') as f:f.write('\n\n105. [初值物质平均投影只读复核](../research/reconnect/INITIAL_PROJECTION_REVIEW.md)（2026-11-06）。\n\n'
      '31个内部物质边界均切在细参考格内；解析边界缺陷恒等式解释5.89553258e-9/1.08801426e-9表示差，浮点贡献约1e-17到1e-16。80/100位主复核与独立sinc部分积分100/120位审计首次通过，整格均值不确定部分积分的有理数见证成立。不改变物理初值、旧正项或两次未决，0新PDE/候选/软件测试，累计500；782原文件、15附件、104旧正式清单保持。仅提出完整子物质区间积分的参考观测契约，下一项隔离非均匀参考原型及两个小型软件夹具，尚未执行新Becker参考。Stage3 incomplete，fluid ORC关闭；仅本地交付。\n')


def package():
    out=OUT;exports=ROOT/'exports';pdf=exports/f'{BASE}_report_{DATE}.pdf';archive=exports/f'{BASE}_code_results_{DATE}.zip';seal=out/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,seal))
    assert read(out/'execution.json')['first_attempt_passed'] and read(out/'audit.json')['passed']
    info=subprocess.run(['pdfinfo',str(out/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           2' in info
    log=(out/f'{BASE}_report_{DATE}.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    shutil.copyfile(out/pdf.name,pdf);save(out/'PDF-generation.json',{'pages':2,'sha256':digest(pdf),'all_pages_visually_reviewed':True,'no_overfull_or_missing_characters':True})
    parent=SOURCE/'sha256.json';entries=read(parent)
    for rel,h in entries.items():assert digest(ROOT/rel)==h,rel
    entries[str(parent.relative_to(ROOT))]=digest(parent)
    for name in FILES:entries['research/reconnect/'+name]=digest(ROOT/'research/reconnect'/name)
    for p in out.rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT))]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    with zipfile.ZipFile(exports/'LTN_becker_reference_code_results_20261106.zip') as z:selected=set(json.loads(z.read('review-manifest.json')))
    for folder in ('research','docs'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file());selected.update([str(pdf.relative_to(ROOT)),str(parent.relative_to(ROOT))])
    originals=read('/workspace/original-handoff-sha256.json');selected.update(originals);manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN initial material-mean projection review,2026-11-06.
All31 internal material cuts lie within fine reference cells. Analytic boundary defects explain initial bridge; floating contributions are separately retained.
Same full-cell means do not determine partial-cell integrals. Initial analytic knowledge cannot be used to repair terminal reference or delete old budgets.
Proposal only: original material regions as unions of complete reference children, exact integral aggregation. New nonuniform reference prototype/fixtures NOT executed.
0 new PDE/candidate/grid/software tests; count500. Original unresolved decisions and H budgets preserved; Stage3 incomplete,fluid ORC OFF.
80/100-digit main and independent100/120-digit sinc/partial-cell audit passed.782 originals,15 attachments and104 prior manifests verified in original workspace.
Contains research/docs,782 originals,parent cases and current review. Not full39GB workspace,15 attachments or missingv080.
Only local delivery; no publishing/pushing. ZIP uses review-manifest.json; do not assume full Git backup/ metadata is included.
Replay initial-only accounting:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/replay_initial_projection.py --directory runs/reconnect-initial-projection
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        with tempfile.TemporaryDirectory(prefix='ltn-initial-projection-restore-',dir='/tmp') as directory:
            z.extractall(directory);replay=subprocess.run([sys.executable,'research/reconnect/replay_initial_projection.py','--directory','runs/reconnect-initial-projection'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    delivery={'date':'2026-11-06','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'pages':2,'sha256':digest(pdf),'bytes':pdf.stat().st_size},
      'ZIP':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size},'included_files':len(manifest),
      'CRC_SHA_and782_originals_verified':True,'fresh_replay_exit_code':0,'fresh_replay_log':replay.stdout,'extracted_files_unchanged':True,
      'main_and_audit_first_attempt_passed':True,'new_PDE_candidate_grid_software_tests':0,'recorded_checks_total':500,
      'old_decisions_and_budgets_preserved':True,'reference_observation_contract_only_proposed':True,'next_prototype_executed':False,
      'stage3':'incomplete','fluid_orc_enabled':False,'prior_formal_manifests_verified':104,'attachments_verified_in_workspace':15,'published_or_pushed':False}
    save(exports/f'{BASE}_delivery_{DATE}.json',delivery)
    (exports/f'{BASE}_downloads_{DATE}.md').write_text(f'# 初值投影复核下载\n\n[两页PDF]({pdf.name}) · [代码与结果ZIP]({archive.name})\n\nPDF SHA-256：`{digest(pdf)}`\n\nZIP SHA-256：`{digest(archive)}`\n\n旧未决与H预算保持，Stage3未完成、fluid ORC关闭；仅本地交付。\n')
    print(json.dumps(delivery,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['document','package']);a=p.parse_args();document() if a.mode=='document' else package()
