# 移动Becker保存H预算复核：参考空间差主导，原单例仍未决

2026-11-04（Asia/Shanghai）。执行[原登记只读项](../../runs/reconnect-becker-point/next-gate-spec.json)，先冻结[复核协议](becker_budget_review_protocol.json)，再读取既存文件。[原单例报告](BECKER_POINT_COMPARISON.md)及全部原始数据保持。

**结论：saved_reference_space_dominates_reference_only_check_justified。** 这是保存预算的来源判断，不是移动Becker物理通过。原 `moving_Becker_control_failed_or_unresolved` 决策、H公式、阈值和初始分母保持；Stage3 incomplete，fluid ORC关闭，κ未计算。参考空间差主导预算，不代表它造成了全部物理误差。

## 原公式与正项账目

令 b 为原固定正项和，A=2√M·max(1,max|c0|)，E0为原初始质量加权方差。原指标严格保持 **h(b)=(Ab+b²)/E0**；此处 E0=0.1138133441900336，A=3.81971076560784。原 b=1.4475483566214639e-07、h(b)=4.8581440870698443e-06；实际H差的下降仅 5.7627959992867517e-07。

| 正项分量 | 绝对状态预算 | 占原状态预算 | 保存来源（相对本例source/） |
|---|---:|---:|---|
| 两倍参考空间差 | 1.44492358205e-07 | 99.818674% | source-column-inputs.npz: reference_n4096-k4096 minus reference_n8192-k4096 |
| 两倍参考时间差 | 1.07086300522e-10 | 0.073977702% | source-column-inputs.npz: reference_n8192-k4096 minus reference_n8192-k8192 |
| 两倍三路径机器指标 | 1.55385466832e-10 | 0.10734389% | k4,k8,k16/native-results.json: final cumulative_machine_indicator_absolute |
| 两倍三路径尾项 | 1.94296518544e-15 | 1.3422454e-06% | k4,k8,k16/native-results.json: final cumulative_tail_indicator_absolute |
| 两倍完整独立序列观测差 | 3.0457937728e-15 | 2.104105e-06% | k4,k8,k16/trajectory-states.npz: final actual minus complete independent state |
| 原初值观察桥 | 1.39929582254e-16 | 9.6666603e-08% | results.json: time_control.components_absolute.original_column_initial_observation_bridge |
| K4观察舍入桥 | 2.34830641681e-16 | 1.6222646e-07% | results.json: time_control.components_absolute.k4_column_observation_rounding_bridge |
| K8观察舍入桥 | 1.88020984374e-16 | 1.2988926e-07% | results.json: time_control.components_absolute.k8_column_observation_rounding_bridge |
| K16观察舍入桥 | 1.38100404793e-16 | 9.5402965e-08% | results.json: time_control.components_absolute.k16_column_observation_rounding_bridge |

占比属于状态预算b；各分量单独代入h所得值不能相加为独立H误差归因。正项精确有理数总和与原浮点累加的差另存 `results.json`，仅保留舍入账目，不用带符号抵消降低判别预算。

## 为何应核验参考，而不是增加候选K

参考空间差在b中占 99.818674%；参考时间差占 0.073978%，机器指标占 0.107344%。仅参考空间分量按原公式得到的诊断指标 4.84933503176e-06 已大于实际H下降。非参考项单独的原公式诊断值 5.21511143976e-09 则小得多。

仅为工作规划，反解**同一个**原多项式 h(b)=已存H下降：总b等值点约1.71710142957e-8；在其他原正项不变时，参考空间+时间指标须小于约 8.50781156959e-09，约为现值的1/8.49804。这是原判据的可分辨余量分析，不是新的PDE精度阈值，不能给旧结果补发通过。

独立精确重叠积分与原保存32列参考的质量范数差 1.2490742834e-14，只核验保存投影，没有替换原均值或添加到旧预算。已存参考的H空间差 6.49469662495e-08、H时间差 5.85947956822e-11 仅为观察，不用于事后收紧原公式。参考细化差均为经验指标，**不是认证连续PDE误差界**。

点密度/真物质平均密度、点Voronoi/真物质体积、m/点密度与真体积的最大相对差仍分别约8.26%、21.42%、7.63%。这些来自旧诊断，不改点、不修体积、不分摊因果误差。

## 下一项已登记，尚未运行

[固定参考解L形核验](../../runs/reconnect-becker-budget-review/next-gate-spec.json)：只运行独立参考求解器的三点 (Nref,Kref)=(16384,8192)、(32768,8192)、(32768,16384)，分别取得同一时间精度的空间差和同一空间精度的时间差。**候选仍为原32物质列、8行、K4/8/16；0新图、0候选传播、0旧路径重跑。** 原中心通量、状态平均和所有候选数据不动。

选择依据仅为参考中心空间离散的形式二阶成本规划：只加一个参考空间档时，预计空间差缩4倍仍不足；两档使预计相邻空间差缩16倍，才有分辨余地。该估计不认证阶或误差、不合成Richardson状态，也不保证下一项通过。原8192×8192参考耗时5.709秒，按Nref×Kref线性成本约79.9秒规划，300秒/1024MiB封顶。

核验须独立精确质量重叠投影，保留参考极值/质量/非有限数守卫；所有空间、时间、投影和初始桥项取正，不拟合、不裁剪、不归一化。只加载旧候选终态，按原H公式作**另行登记的新比较**；旧未决始终保留。若失败或仍未分辨即停止，不自动增加参考/候选N或K，也不能据此单独宣告移动激波T4或Stage3完成。

## 执行、审计与交付

本轮0新图、0候选作用、0新PDE、0新状态、0新软件测试，累计仍500项。只读主复核首次成功，22.483秒/62.01MiB；独立标准库精确有理数和80/100位Decimal账目审计首次通过，21.424秒。782原始文件、15附件、102份旧正式清单及4份历史诊断差异保持。原主计算末尾的诊断失败与109个补全前文件保持，未把首次失败改成成功。

本轮仅本地交付，不发布环境或软件版本；公开GitHub推送仍待明确公开授权。PDF和ZIP的CRC/SHA及解压复核以交付JSON为准；ZIP包含研究代码/文档、782原文件、原移动Becker完整可重放结果和本轮只读复核，不含39GB全工作区、15附件或未取回v080。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python research/reconnect/replay_becker_budget_review.py --directory runs/reconnect-becker-budget-review
```
