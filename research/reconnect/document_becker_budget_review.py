"""Source-linked saved-budget report; no numerical experiment or reference solve."""
from pathlib import Path
import shutil
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-becker-budget-review';r=read(out/'results.json');ex=read(out/'execution.json');audit=read(out/'accounting-audit.json')
    assert ex['review_first_attempt_passed'] and audit['audit_first_attempt_passed']
    for name in ('main','audit'):shutil.copyfile(f'/tmp/ltn-becker-budget-review-{name}.log',out/f'{name}.log')
    titles={'twice_reference_space':'两倍参考空间差','twice_reference_time':'两倍参考时间差','twice_machine':'两倍三路径机器指标',
      'twice_tail':'两倍三路径尾项','twice_observed_full_sequence':'两倍完整独立序列观测差',
      'original_column_initial_observation_bridge':'原初值观察桥','k4_column_observation_rounding_bridge':'K4观察舍入桥',
      'k8_column_observation_rounding_bridge':'K8观察舍入桥','k16_column_observation_rounding_bridge':'K16观察舍入桥'}
    table='\n'.join(f"| {titles[v['component']]} | {v['absolute']:.12g} | {100*v['fraction_of_original_state_budget']:.8g}% | {v['source']} |" for v in r['components'])
    ceiling=float(r['feasibility']['same_formula_reference_space_time_ceiling']);factor=float(r['feasibility']['saved_reference_space_time_over_ceiling'])
    report=f'''# 移动Becker保存H预算复核：参考空间差主导，原单例仍未决

2026-11-04（Asia/Shanghai）。执行[原登记只读项](../../runs/reconnect-becker-point/next-gate-spec.json)，先冻结[复核协议](becker_budget_review_protocol.json)，再读取既存文件。[原单例报告](BECKER_POINT_COMPARISON.md)及全部原始数据保持。

**结论：saved_reference_space_dominates_reference_only_check_justified。** 这是保存预算的来源判断，不是移动Becker物理通过。原 `moving_Becker_control_failed_or_unresolved` 决策、H公式、阈值和初始分母保持；Stage3 incomplete，fluid ORC关闭，κ未计算。参考空间差主导预算，不代表它造成了全部物理误差。

## 原公式与正项账目

令 b 为原固定正项和，A=2√M·max(1,max|c0|)，E0为原初始质量加权方差。原指标严格保持 **h(b)=(Ab+b²)/E0**；此处 E0={r['original_initial_variance_binary64']:.17g}，A={r['original_H_prefactor']:.17g}。原 b={r['original_binary64_budget_absolute']:.17g}、h(b)={r['original_binary64_H_indicator']:.17g}；实际H差的下降仅 {r['saved_H_decline']:.17g}。

| 正项分量 | 绝对状态预算 | 占原状态预算 | 保存来源（相对本例source/） |
|---|---:|---:|---|
{table}

占比属于状态预算b；各分量单独代入h所得值不能相加为独立H误差归因。正项精确有理数总和与原浮点累加的差另存 `results.json`，仅保留舍入账目，不用带符号抵消降低判别预算。

## 为何应核验参考，而不是增加候选K

参考空间差在b中占 {100*r['components'][0]['fraction_of_original_state_budget']:.6f}%；参考时间差占 {100*r['components'][1]['fraction_of_original_state_budget']:.6f}%，机器指标占 {100*r['components'][2]['fraction_of_original_state_budget']:.6f}%。仅参考空间分量按原公式得到的诊断指标 {r['same_formula_reference_space_only_floor_diagnostic']:.12g} 已大于实际H下降。非参考项单独的原公式诊断值 {r['same_formula_nonreference_only_floor_diagnostic']:.12g} 则小得多。

仅为工作规划，反解**同一个**原多项式 h(b)=已存H下降：总b等值点约1.71710142957e-8；在其他原正项不变时，参考空间+时间指标须小于约 {ceiling:.12g}，约为现值的1/{factor:.6g}。这是原判据的可分辨余量分析，不是新的PDE精度阈值，不能给旧结果补发通过。

独立精确重叠积分与原保存32列参考的质量范数差 {r['projection']['independent_exact_overlap_vs_saved_means_absolute']:.12g}，只核验保存投影，没有替换原均值或添加到旧预算。已存参考的H空间差 {r['saved_reference_H_space_difference_diagnostic']:.12g}、H时间差 {r['saved_reference_H_time_difference_diagnostic']:.12g} 仅为观察，不用于事后收紧原公式。参考细化差均为经验指标，**不是认证连续PDE误差界**。

点密度/真物质平均密度、点Voronoi/真物质体积、m/点密度与真体积的最大相对差仍分别约8.26%、21.42%、7.63%。这些来自旧诊断，不改点、不修体积、不分摊因果误差。

## 下一项已登记，尚未运行

[固定参考解L形核验](../../runs/reconnect-becker-budget-review/next-gate-spec.json)：只运行独立参考求解器的三点 (Nref,Kref)=(16384,8192)、(32768,8192)、(32768,16384)，分别取得同一时间精度的空间差和同一空间精度的时间差。**候选仍为原32物质列、8行、K4/8/16；0新图、0候选传播、0旧路径重跑。** 原中心通量、状态平均和所有候选数据不动。

选择依据仅为参考中心空间离散的形式二阶成本规划：只加一个参考空间档时，预计空间差缩4倍仍不足；两档使预计相邻空间差缩16倍，才有分辨余地。该估计不认证阶或误差、不合成Richardson状态，也不保证下一项通过。原8192×8192参考耗时5.709秒，按Nref×Kref线性成本约79.9秒规划，300秒/1024MiB封顶。

核验须独立精确质量重叠投影，保留参考极值/质量/非有限数守卫；所有空间、时间、投影和初始桥项取正，不拟合、不裁剪、不归一化。只加载旧候选终态，按原H公式作**另行登记的新比较**；旧未决始终保留。若失败或仍未分辨即停止，不自动增加参考/候选N或K，也不能据此单独宣告移动激波T4或Stage3完成。

## 执行、审计与交付

本轮0新图、0候选作用、0新PDE、0新状态、0新软件测试，累计仍500项。只读主复核首次成功，{ex['elapsed_main_seconds']:.3f}秒/{ex['peak_RSS_MiB']:.2f}MiB；独立标准库精确有理数和80/100位Decimal账目审计首次通过，{audit['elapsed_seconds']:.3f}秒。782原始文件、15附件、102份旧正式清单及4份历史诊断差异保持。原主计算末尾的诊断失败与109个补全前文件保持，未把首次失败改成成功。

本轮仅本地交付，不发布环境或软件版本；公开GitHub推送仍待明确公开授权。PDF和ZIP的CRC/SHA及解压复核以交付JSON为准；ZIP包含研究代码/文档、782原文件、原移动Becker完整可重放结果和本轮只读复核，不含39GB全工作区、15附件或未取回v080。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
python research/reconnect/replay_becker_budget_review.py --directory runs/reconnect-becker-budget-review
```
'''
    path=ROOT/'research/reconnect/BECKER_BUDGET_REVIEW.md';assert not path.exists();path.write_text(report)
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}\setmainfont{Noto Serif CJK SC}\setsansfont{Noto Sans CJK SC}\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
\usepackage[margin=21mm]{geometry}\usepackage{amsmath,booktabs,hyperref,fancyhdr}
\hypersetup{colorlinks=true,urlcolor=blue}\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small diffuse-mixing / LTN}\fancyhead[R]{\small 2026-11-04}\fancyfoot[C]{\thepage}
\begin{document}
\begin{center}{\LARGE\bfseries 移动Becker：保存H预算复核}\\[5pt]
{\large 参考空间差主导；原单例仍未决}\\[4pt]2026年11月4日（Asia/Shanghai）\end{center}
\section*{1. 本轮得到的判断}
\textbf{保持原未决结果，Stage3 incomplete，fluid ORC关闭。}
本轮只读取保存文件，0新图、0候选作用、0新PDE、0新状态、0新软件测试，累计仍500项。
候选为原32物质列、8行、K4/8/16，固定质量平均与所有原数据保持。

原固定指标为
\[h(b)=\frac{A b+b^2}{E_0},\qquad A=2\sqrt{M}\max(1,\max|c_0|).\]
原H下降仅5.762796$\times10^{-7}$，低于原指标4.858144$\times10^{-6}$。
下表分解的是状态预算$b$，不是物理误差的因果分摊：
\begin{center}\begin{tabular}{lrr}\toprule
保存的正项&绝对状态预算&占原$b$\\\midrule
参考空间差（两倍）&1.4449235821e-7&99.818674\%\\
参考时间差（两倍）&1.0708630052e-10&0.073978\%\\
三路径机器指标（两倍）&1.5538546683e-10&0.107344\%\\
三路径尾项（两倍）&1.9429651854e-15&0.00000134\%\\
完整独立序列差（两倍）&3.0457937728e-15&0.00000210\%\\
原初值与三观察桥合计&7.0088161306e-16&0.00000048\%\\\bottomrule
\end{tabular}\end{center}
各正项和与原浮点累加的舍入差另存精确有理数账目；不以带符号抵消减小预算。
各分量单独代入$h$所得值不能直接相加为独立H误差贡献。

\section*{2. 下一步应核验参考精度}
非参考项按原公式的诊断值约5.215111$\times10^{-9}$，小于已存H下降。
仅参考空间项的原公式诊断值约4.849335$\times10^{-6}$，已大于下降。
因此增加候选K不能消除当前预算主项。

只为工作规划，反解\emph{同一个原公式}在已存H下降处的等值点，得总$b$约
1.71710143$\times10^{-8}$。其他原正项不变时，参考空间加时间指标需低于约
8.50781157$\times10^{-9}$，即比现值小约8.50倍。
\textbf{这不是新PDE精度阈值，不能追认旧结果通过。}

独立精确重叠积分与保存32列参考的范数差为1.24907428$\times10^{-14}$；
它核验投影，不认证连续PDE参考误差。保存参考空间与时间差分别约
7.22461791$\times10^{-8}$、5.35431503$\times10^{-11}$，均为经验指标。
不以保存参考的直接H差或更紧后验公式替代原判据。
\newpage
\section*{3. 仅登记一次固定参考核验，尚未执行}
原独立细质量网格参考采用共享中心面通量及时间中点Crank--Nicolson，
不导入候选有限体积后端。下一项固定为下列L形三点：
\begin{center}\begin{tabular}{rrl}\toprule
$N_{\rm ref}$&$K_{\rm ref}$&目的\\\midrule
16384&8192&同时间精度的空间比较\\
32768&8192&分开空间差与时间差\\
32768&16384&同空间精度的时间比较\\\bottomrule
\end{tabular}\end{center}
\textbf{0新候选图、0候选传播、0旧路径重跑。} 原候选N、K、行数、点中心和物理参数不变。
参考空间、时间差及新增投影/初值桥取正，保留原机器/尾项/独立序列/观察桥。
新比较只读旧终态，使用原质量、初值、波动分母和H公式；旧未决文件保持。

选择两档参考空间仅依据中心离散形式二阶作成本规划：一档预计缩4倍仍不足，
两档预计相邻差缩16倍才有余地。\textbf{预计不等于实测阶或误差保证，
不合成Richardson状态，也不保证通过。}
8192$\times$8192旧参考耗时5.709秒；按网格乘时间步规划三点约79.9秒，
300秒/1024MiB封顶。若失败或仍未分辨，保持原结果并停止，不自动增加参考或候选N/K。

参考须检查原质量漂移、初始极值及非有限数守卫，独立精确质量重叠投影至原32列；
不裁剪、归一化、拟合F/C、调整阈值或事后收紧H公式。
即使新保存状态/参考比较分辨，也不能单独宣告移动激波T4、三密度来源或Stage3完成。

\section*{4. 保存诊断与审计范围}
点密度与真物质平均密度、点Voronoi体积与真物质体积、$m/\rho_{\rm point}$与真体积
最大相对差仍约8.26\%、21.42\%、7.63\%。这是旧保存诊断，不修改代表点、修补体积或分摊因果误差。

只读主复核首次成功；独立标准库精确有理数及80/100位Decimal账目审计首次通过。
782原始文件、15附件、102份旧正式清单和4份历史诊断差异保持。
原科学计算末尾诊断解析失败与109个补全前文件保持，不能改写成首次成功。

\section*{5. 交付与恢复}
代码ZIP包含研究代码/文档、782原始文件、移动Becker完整可重放结果及本轮复核。
不是39GB全工作区镜像，不含15附件或未取回v080。逐文件SHA、CRC及新解压复核以交付JSON为准。
本轮仅本地交付，不发布环境或软件版本；公开GitHub推送仍待明确公开授权。

完整来源表见BECKER\_BUDGET\_REVIEW.md与budget-components.csv。
保存数据复核入口：
\begin{verbatim}
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 python \
research/reconnect/replay_becker_budget_review.py \
--directory runs/reconnect-becker-budget-review
\end{verbatim}
\end{document}
'''
    (out/'LTN_becker_budget_report_20261104.tex').write_text(tex)
    status=ROOT/'docs/RESEARCH_STATUS.md';text=status.read_text();assert '\n103. ' not in text
    with status.open('a') as f:f.write(f'\n\n103. [移动Becker保存H预算只读复核](../research/reconnect/BECKER_BUDGET_REVIEW.md)（2026-11-04）。\n\n'
      f'保持原未决/H公式与阈值；原状态预算99.818674%来自参考空间差，非参考项单独原H诊断值5.21511144e-9。按同一公式反解，参考空间+时间指标约需缩小8.50倍才有分辨余地。此为工作规划，不追认旧通过、不认证PDE误差界。0新图/候选作用/PDE/状态/软件测试，累计仍500；主复核与独立80/100位账目审计首次通过。782原文件、15附件、102旧正式清单和历史诊断差异保持。\n\n'
      '仅登记独立参考L形三点(16384,8192)/(32768,8192)/(32768,16384)，候选N/K/行数/中心/参数不变，尚未执行。保留参考空间/时间/投影正项与原H公式；失败或未决即停，不自动加密或改阈值。Stage3 incomplete，fluid ORC关闭；本轮本地PDF/代码ZIP交付，公开推送待明确授权，不发布环境/版本。\n')
    print('Source-linked saved-budget Markdown, two-page LaTeX and status103 written; future reference check not executed.',flush=True)


if __name__=='__main__':main()
