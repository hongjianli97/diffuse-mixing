# 光滑压缩映射：同质量同初态的非均匀点云冻结检验

2026-11-01，Asia/Shanghai。Stage 3 **incomplete**，fluid ORC **关闭**。

本轮执行上一轮预定的单一光滑压缩映射控制，保留原m、c0、身份、N=128、h=0.04和D=0.02。冻结时刻的瞬时质量加权变化率误差为 **0.896770%**。图层保持非负、守恒；解析材料面通量与独立Gauss参考差约3.24e−15。rho=1来自已给定运动的rho0/J恒等式，不是覆写密度数组；当前点云仍非均匀。

这是一个新增物理状态，不与上一轮平移流的有限时间浓度误差或不同状态的变化率误差作改进百分比比较。尚未分开本例有限核宽、非均匀求积及平均状态闭合误差；不能把0.8968%全部称为几何误差。没有运行压缩流新瞬态或ORC，没有调整权重、移动节点或改阈值。

## 1. 固定材料与给定运动

核验 `runs/reconnect-kernel2d-density-transient/` 的563项科学清单及其 `next_compression_probe_predeclared`。直接读取保存的m、c0、ids、物质标签中心及初始点rho0，不用上一轮演化后的浓度替代c0。当前冻结标量仍是同一个预先指定材料标签场的质量平均，不宣称已在新压缩流中扩散演化。

给定周期运动为

\[
X_x=a_x+\frac{\alpha(t)}{2\pi}\sin(2\pi a_x)+0.2t,
\quad X_y=a_y-0.1t\pmod1,
\quad \alpha(t)=0.25t/0.05,
\]

时间记录只有显式定义的0、0.05 s，冻结第二帧。rho0(a)=1+0.25 cos(2pi a_x)，解析

\[
J=1+\alpha\cos(2\pi a_x)>0,\qquad \rho=\rho_0/J.
\]

J只用于解析轨迹/密度生产和独立参考，没有进入点核接口，也未从点云拟合F/C。候选仍只接收 `TrajectorySamples` 中的位置、固定质量和oracle点密度。

在预定alpha=epsilon=0.25时rho=1，真实当前面积等于原固定质量（单位rho和单位厚度）。特意使用这一恒等关系来排除密度重构/体积权重误差的混淆，不扫描alpha寻找最小误差，也不将这个特殊结果外推为一般压缩流密度闭合证明。该运动是连续性基准，不称为无外力Navier–Stokes解。

## 2. 质量与几何审计

代码逐点计算rho0/J，不将冻结密度设为常数。保存的J采样范围[0.7500753,1.2499247]，理论全域下界0.75；rho*J−rho0最大差为零，材料连续性D_t rho+rho div u残差8.88e−16。m、c0、身份保持原样。

用不取模的真实物质端点计算每个方格当前面积，避免周期截口导致错误负体积。另以解析面积Delta²[1+alpha sinc(1/N)cos(2pi a_x)]独立交叉检查。

| 几何/权重诊断 | 数值 |
|---|---:|
| 冻结点rho范围，kg/m³ | [1,1] |
| 保存m相对真实当前面积最大差 | 1.58e−14 |
| 原始m/rho_point求积面积相对真实面积最大差 | 1.58e−14 |
| 真实面积：面差与解析式最大相对差 | 1.57e−14 |
| 原始求积面积总和、真实面积总和 | 均为1 m² |
| 真实流向单元宽度范围 | [0.00586016,0.00976484] m |
| 轨迹点相对真实几何中心最大偏移 | **1.19800e−5 m** |

面积差属于端点相减的浮点核对尺度，没有通过归一化消除。轨迹点仍为X(a_mid,t)，非仿射运动后不必等于两真实面之间的几何中点。偏移原样保留，不把点移到几何中心来改善结果。候选没有读取这里的真实面积、面或中心。

## 3. 独立完整PDE质量平均参考

物理连续目标仍为rho D_t c=div(rho D grad c)。在当前映射中，材料散度形式为

\[
\rho_0 r=D\partial_a(\rho_0J^{-2}c_a)+D\rho_0c_{bb}.
\]

定义材料扩散梯度通量phi(a)=D*rho0/J²*c_a。纵向参考由phi在真实物质标签左右端点之差积分；横向参考为−D(2pi)²(cbar−0.5)。分母保留保存的原m，不以新几何估计重建质量。

独立Gauss8/16在物质区域积分rho0乘以完整物理点PDE，其中

\[
\Delta_xc=c_{aa}/J^2-c_aJ_a/J^3+c_{bb},
\]

\[
\nabla_x\log\rho\cdot\nabla_xc=
(\rho_{0,a}/\rho_0-J_a/J)c_a/J^2.
\]

本冻结时刻漂移项因rho真实均匀而为零；通用参考公式仍保留它，并在非匹配alpha的单元检查中与独立复步材料通量导数核对。没有为得到零漂移而删除项。

| 独立参考核对 | 数值 |
|---|---:|
| 解析面/Gauss16变化率相对差 | 3.24e−15 |
| 解析面/Gauss16变化率最大差 | 2.11e−15/s |
| Gauss8/Gauss16变化率相对差 | 2.07e−16 |
| 保存m/Gauss16质量最大相对差 | 4.26e−16 |
| 保存c0/Gauss16质量平均最大差 | 4.44e−16 |

加密差为经验核对，不称为严格误差界。

## 4. 候选实际作用与结论范围

继续使用原二维C4点核、调和rho*D、固定h和完整周期支撑。当前16384节点、2788224条非零共享边，图连通，最小正电导约1.95e−28 kg/s，未截断这些小权重。原始Q最小非对角为零。

\[
e_r=\frac{\sqrt{\sum_i m_i(r_i^{kernel}-r_i^{PDE})^2}}
{\sqrt{\sum_i m_i(r_i^{PDE})^2}}=0.896770\%.
\]

最大绝对变化率误差0.00607675/s。常数残差6.33e−14/s，质量残差1.01e−17 kg/s，加权对称性残差5.42e−20。独立共享边组装与候选L差1.73e−17，边通量散射与稀疏实际作用差7.12e−14/s。峰值RSS约633.5 MiB，没有N×N稠密矩阵或P。

这一控制说明，在当前给定运动与分辨率下，真实质量/体积关系和非均匀位置没有破坏原始正交换图，独立物理误差已量化。它仍混合有限h模型、非均匀求积、节点非几何中心及平均状态闭合影响；不能仅因权重面积准确就宣称连续一致性。物理标量在空间中的形状也随映射改变，因此不能把它与平移流的误差差值归为纯几何效应。

![压缩点云、密度恒等关系及原始节点偏移](../../runs/reconnect-kernel2d-compression/compression.png)

## 5. 检查与保存

最终8项新增检查全部通过：rho0/J连续性及特殊均匀密度恒等式、材料面/独立Eulerian Gauss、复步材料通量散度、零变形退化到原平移基准、真实面积和中心偏移、非均匀点图正性/D=0、映射导数与周期性、非法映射/时间外推拒绝。初次一项连续性检查的预期数组未展开时间维，修正测试形状后通过；绝对容差和模型不改，初次日志保留。按最终8项计数，累计272+8=280，不是旧246项恢复。

数据位于 `runs/reconnect-kernel2d-compression/`，含原m/c0/身份、压缩位置、按rho0/J生成的oracle密度、参考J/面/中心、完整共享边/CSR、解析与Gauss参照及原始误差。核验782个原始文件、15个历史附件及31份已有正式科学清单不变；4份历史诊断清单既有路径差异另保留。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_compression_probe.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/compression_study.py --output /workspace/compression-frozen-reproduction
```

这是一个特殊冻结态，不是一般非均匀点云、变密度/几何耦合、时变传播、激波/接触、固壁、ledger、非平面或DNS验收。Stage 3保持incomplete，fluid ORC关闭。未生成PDF/版本，未提交、推送、修改环境配置或发布。

## 6. 下一步只补同一压缩轨迹的中点

保持同一运动、N/h/D、原m/c0，只新增t=0.025 s的冻结检验，此时alpha=0.125、rho=rho0/J不再均匀。使用同一通用材料散度参考及Gauss8/16，记录原始m/rho_point面积偏差，检验非均匀点几何与密度漂移叠加时的实际作用。不另选压缩幅值，不扫描时间寻找较优结果，不运行压缩流瞬态。

另复用已保存的起点（alpha=0，平移不改材料距离）和终点算子，仅计算在原c0上的嵌套作用[Q_end,Q_start]c0，量化实际流动下的非对易性。不得构造全矩阵乘积、平均生成元传播子或ORC；该诊断不是新时间推进。它用于判断下一步实际时变传播必须保留的时间顺序，而不是为图层性质补拟合权重。

## 7. 交付

- [小型审阅包](../../exports/ltn-compression-frozen-review.zip)：代码、报告、图表、日志、审计及小型物理观测数组。
- [完整科学数据包](../../exports/ltn-compression-frozen-full.zip)：本轮原始共享边/CSR及引用的历史科学数据；不重复所有原始附件。
- [完整分卷下载目录](../../exports/ltn-compression-frozen-downloads.md)、[交付哈希](../../exports/ltn-compression-frozen-delivery.json)及[重组脚本](../../exports/restore-compression-frozen.py)。分卷按原ZIP字节切分，先重组再解压。原782文件审计跨环境仍需原交接材料。
