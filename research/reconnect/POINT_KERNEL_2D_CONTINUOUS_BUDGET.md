# 同一光滑变密度状态：连续核参照与三项误差预算

2026-10-31。Stage 3 **incomplete**，fluid ORC **关闭**。

本轮不改变已有N=64、h=0.06、D=0.02、rho、质量平均状态或离散算子，仅计算上一轮预定的两种连续参照。保存的 **1.620938%** 变化率误差中，物理有限核宽偏差的相对范数为 **1.613911%**，平均输入/点密度闭合差为 **0.017336%**，节点求积差为 **0.00078494%**。两组参照积分的差远小于各分项，支持在本状态中将有限核宽识别为主要误差来源。

这些是三个有符号向量的范数，不能直接相加；保存了Gram交叉项。没有新建图、重算离散作用、推进瞬态或修改权重。总误差没有在本轮降低，本轮完成的是可核对的来源分离。

## 1. 固定输入与两种参照

直接读取 `runs/reconnect-kernel2d-variable-density/variable-density-kernel2d.npz` 的m、cbar、rho_point、局部PDE参考和已有离散变化率。运行前核对其497项科学清单；协议来自该轮 `next_reference_predeclared`。候选电导代码及所有旧结果保留不变。

物质标签上的rho=1+epsilon cos(kx)，c=0.5+A cos(kx)cos(ky)，epsilon=0.25、A=0.2、k=2pi；仍是单位周期方形上的平移运动冻结状态，单位厚度为1 m。Gamma=rho D，H为调和平均，B_h为已归一化二维C4核的−Wprime/r。

第一种参照是在真实连续标量上施加物理连续核：

\[
\mathcal L_h c(x,y)=\frac2{\rho(x)}\int
H(\Gamma(x),\Gamma(x+u))B_h(\sqrt{u^2+v^2})
[c(x+u,y+v)-c(x,y)]\,du\,dv.
\]

随后对真实物质方格作rho加权质量平均，记为r_h^phys。这一参照包含有限核宽物理模型偏差，却不使用离散节点求积，也不把单元平均当作点值。

第二种参照从本轮已知解析输入精确延拓保存的单元平均与求积权重，而非从离散算子输出拟合。令Delta=1/N，s=sinc(1/N)、s2=sinc(2/N)，

\[
\rho_p(x)=1+\varepsilon\cos(kx),\quad
\bar\rho_\Delta(x)=1+\varepsilon s\cos(kx),\quad
w_\Delta(x)=\bar\rho_\Delta(x)/\rho_p(x),
\]

\[
f_\Delta(x)=A s\frac{s\cos(kx)+\frac\varepsilon2[1+s_2\cos(2kx)]}
{\bar\rho_\Delta(x)}.
\]

保存状态在中心满足cbar=0.5+f_Delta(x)cos(ky)，求积面积为Delta²*w_Delta(x)。将这些固定函数代入连续求积，得到r_h^input：

\[
r_h^{input}(x,y)=\frac2{\rho_p(x)}\int w_\Delta(x+u)
H(\Gamma(x),\Gamma(x+u))B_h(|(u,v)|)
[f_\Delta(x+u)\cos(kv)-f_\Delta(x)]\,du\,dv\;\cos(ky).
\]

中心前因子2/rho_point来自原离散式的2*Aq_i/m_i，原始质量和权重没有被重标定。第二种参照仍保留固定Delta的平均状态、点密度及求积闭合，仅将节点求和换成连续积分。

延拓在保存中心与原状态的最大差1.11e−16，点rho、固定质量与权重比逐项差均为零。该延拓由这个解析输入唯一指定；不将它当成任意稀疏轨迹的通用重构定理。

## 2. 降维、积分与独立验证

rho只随x变化，标量沿y是单一余弦。v的奇项积分为零，物理核可写成F_h(x)cos(ky)。物理单元平均通过x向rho加权积分与精确y平均s cos(ky_center)计算，不遗漏平均密度分母。

邻域积分采用极坐标u=r cos(theta)、v=r sin(theta)，保留Jacobian r，r在[0,2h]。按预定的64个独立x中心计算，不生成新粒子或图。两组积分严格为：

| 参照组 | 径向Gauss阶数 | 周向周期梯形节点 | 单元x向Gauss阶数 |
|---|---:|---:|---:|
| 原定低阶 | 32 | 64 | 8 |
| 原定高阶 | 64 | 128 | 16 |

高阶结果作为本轮预算参照，低高阶差作为经验数值核对。另验证：均匀rho极限与独立Bessel衰减率一致；二维完整极坐标积分与分离后的公式一致；单点笛卡尔圆盘积分与极坐标积分一致；周期反射、D线性及D=0成立。

独立笛卡尔检查初次使用96×96点，其结果与极坐标相差1.0294e−10/s，未满足原检查精度。加密为192×192后差降至8.10e−13/s，通过原来的十位小数检查，未降低标准。初次失败日志、加密诊断和最终日志一并保留；正式两组极坐标协议不改。这是辅助参照求积精度的修正，没有改变物理参数、候选或验收阈值。

## 3. 固定参照链的三项分解

记保存离散变化率为r_d、完整局部PDE质量平均参考为r_PDE。定义

\[
e_{width}=r_h^{phys}-r_{PDE},\qquad
e_{closure}=r_h^{input}-r_h^{phys},\qquad
e_{quad}=r_d-r_h^{input}.
\]

因此r_d−r_PDE=e_width+e_closure+e_quad。各项统一除以sqrt(sum m*r_PDE²)。

| 范数诊断 | 相对范数 |
|---|---:|
| 保存总误差 | **1.620938%** |
| 物理有限宽偏差 | **1.613911%** |
| 质量平均输入/点密度求积闭合差 | **0.0173355%** |
| 节点求积差 | **0.00078494%** |

有符号向量重构残差为零。三个范数之和约1.63203%，并不等于总误差。归一化Gram矩阵包含全部自项与交叉项；不能将范数分别称作可相加的物理误差百分比。

同时保存各项沿总误差方向的有符号投影占比，依次约99.5616%、0.39123%、0.04716%。这只是上述固定参照链的向量投影，不是概率、拟合权重或唯一的物理因果分摊。

本轮“闭合差”同时包含单元平均状态代入中心积分、点rho与平均rho差异及固定解析延拓的影响，没有将其进一步拟合拆成多个模型。其范数比有限宽项小约两个数量级；继续微调这个小项不会消除当前主要偏差。

## 4. 分离是否超过参照不确定度

| 低高阶差相对局部PDE范数 | 数值 | 差/对应分项范数 |
|---|---:|---:|
| 有限宽分项 | 2.59e−15 | 1.61e−13 |
| 输入闭合分项 | 2.86e−15 | 1.65e−11 |
| 节点求积分项 | 4.26e−16 | 5.43e−11 |

物理连续参照最大点差2.03e−15/s，输入闭合参照最大点差3.05e−16/s。两种参照的质量加权积分均接近零。本例三项都远大于积分核对差，因此可以分辨；但低高阶差仍不是严格误差上界，向量恒等式本身也不能证明参照准确。这里同时有积分加密、均匀Bessel极限和独立二维/笛卡尔积分支撑。

![原始预算与参照积分核对](../../runs/reconnect-kernel2d-continuous-density/continuous-density-budget.png)

## 5. 实现、检查与保留事项

新增 `continuous_density_kernel.py`、`continuous_density_study.py` 及检查/审计/作图脚本。最终7项检查全部通过，初次同7项中有1项失败的日志保留，不重复计数。此前252项，加本轮7项累计记录259项；未重跑全部历史检查，不是旧246项恢复。

结果位于 `runs/reconnect-kernel2d-continuous-density/`：两组完整参照、64点x系数、三个误差向量、Gram和投影、原保存输入/变化率副本、日志及检查结果。没有新图或离散算子作用，进程峰值RSS约74.6 MiB。核验782个原始文件、15个历史附件及28份先前正式科学清单不变；4份历史诊断清单的既有路径差异另保留核对。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_continuous_density.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/continuous_density_study.py --output /workspace/continuous-density-reproduction
```

单个光滑周期状态的参照预算不是变密度收敛证明，更不覆盖激波/接触、非均匀点云、固壁、ledger、非平面或DNS。不能将本轮连续积分用作实际求解器需要的额外输入。fluid ORC关闭；未生成PDF/软件版本、修改环境配置、提交、推送或发布。

## 6. 下一步只做一次变密度联合细化

目前主要偏差已经明确，停止继续拆分较小的输入闭合项。下一项固定N=128、h=0.04，D=0.02、epsilon=0.25、标量及物理运动保持不变，只新增这一档。Delta/h由0.26042降至0.1953125，既缩小核宽又改善节点求积。

复用未修改的正交换核及本轮连续参照定义，检验实际离散误差是否按连续有限宽预测下降。保留稀疏存储，不构造全P或N×N矩阵；不以解析模态修改D/g。两档各与其真实质量平均参考比较；若作同父单元聚合，必须保留旧父质量，不重标定细质量，并单独记录解析质量积分的舍入差。仍不推进瞬态或ORC，也不扩大rho/h/N矩阵扫描。
