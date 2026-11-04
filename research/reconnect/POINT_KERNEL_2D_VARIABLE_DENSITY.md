# 二维点核：一个预定光滑变密度oracle状态

2026-10-31。Stage 3 **incomplete**，fluid ORC **关闭**。

本轮保持二维正交换核代码、N=64、h=0.06、D=0.02、周期方形、平移轨迹及物理标量不变，只执行上一轮预定的rho(a)=1+0.25 cos(2pi a_x)冻结状态。候选对完整变密度PDE质量平均变化率的误差为 **1.620938%**；省略密度漂移的参考与完整PDE之差为 **10.817822%**。原始图非负、守恒，独立参考核对差远小于候选误差。

这支持继续检查该光滑系数核的精度来源，但不等于变密度收敛或Stage 3验收。没有新瞬态、全传播子、ORC、扫描或权重修正。

## 1. 冻结协议与输入来源

读取并核验 `runs/reconnect-kernel2d-refinement/stage3-status.json` 中的 `next_probe_predeclared`。单元为周期单位方形中的64×64规则物质方格，单位厚度b=1 m，Delta=1/64。轨迹x=(a+(0.2,−0.1)t) mod 1，时间帧[0,0.05] s，冻结第二帧。

\[
\rho(a)=1+\varepsilon\cos(k a_x),\quad
c(a)=0.5+A\cos(k a_x)\cos(k a_y),\quad
\varepsilon=0.25,\ A=0.2,\ k=2\pi.
\]

这是密度随匀速平移输运的光滑连续性基准。标量是预先指定的冻结诊断场，不声称已经按扩散从初态演化到0.05 s，也不声称该运动满足无外力的完整Navier–Stokes系统。

固定m由初始物质方格内rho积分给出，cbar为rho加权质量平均。候选只读取位置、固定m和oracle点rho，不读取物质边界、方格连接、F/C、标量公式或密度梯度。真实方格用于输入/独立参考生产，不提升为点核必需接口。

## 2. 解析质量、状态与完整PDE参考

记s=sinc(1/N)、s2=sinc(2/N)，定义每个方格中心的

\[
C=s\cos(k a_x),\quad Y=s\cos(k a_y),\quad
C_2=\tfrac12[1+s_2\cos(2k a_x)].
\]

其中sinc(z)=sin(pi z)/(pi z)。精确密度平均、质量与状态为

\[
\bar\rho=1+\varepsilon C,\quad
m=b\Delta^2\bar\rho,\quad
\bar c=0.5+A\frac{C+\varepsilon C_2}{\bar\rho}Y.
\]

oracle点密度为rho_p=1+epsilon cos(k a_x)，与rho平均严格区分。状态没有改用c(a_i)点值，两者最大差1.60337e−4。

连续目标仍为rho D_t c=div(rho D grad c)。D为常数时

\[
D_t c=D\Delta c+D\nabla\log\rho\cdot\nabla c.
\]

完整质量平均变化率可直接积分散度得到

\[
\bar r=D A k^2\frac{-2C+\varepsilon(1-3C_2)}{\bar\rho}Y.
\]

另保存物理分项

\[
\bar r_{lap}=-2D A k^2\frac{C+\varepsilon C_2}{\bar\rho}Y,\quad
\bar r_{drift}=D A\varepsilon k^2\frac{1-C_2}{\bar\rho}Y.
\]

参考不将局部目标偷换为纯Laplacian。解析散度积分与两分项之和的最大差1.68e−16/s。两分项在这个周期模态上各自也具有近零全局质量加权积分，所以单凭总量守恒检查无法发现漏掉漂移。

独立Gauss8/16直接积分rho、rho*c、rho*(D lap c + D grad log rho·grad c)，不将积分所得m或状态替换回候选输入。另有复步导数对物理通量散度的检查，以及真实单元四面通量积分的检查；这些只用于验证参考。

| 参考核对量 | 数值 |
|---|---:|
| 解析与Gauss16变化率最大差 | 2.22e−16/s |
| 解析/Gauss16变化率相对质量加权差 | 3.51e−16 |
| Gauss8/16变化率相对质量加权差 | 2.21e−16 |
| 解析/Gauss16质量最大相对差 | 3.94e−16 |
| 解析/Gauss16状态最大差 | 2.22e−16 |

这些是独立计算核对尺度，不是对所有误差的严格上界。

## 3. 未修改的点核及求积面积偏差

继续使用上一轮归一化二维C4核、周期完整支撑及共享导通率

\[
A_i^q=m_i/(b\rho_{p,i}),\quad
g_{ij}=2A_i^qA_j^q b\,H(\rho_{p,i}D,\rho_{p,j}D)B_h(r_{ij}).
\]

本例A_i^q=Delta²*rho_average/rho_point，已经不等于真实方格面积。该偏差显式保存，未改变m、点rho或g去消除它。

| 输入或求积量 | 结果 |
|---|---:|
| oracle点rho范围，kg/m³ | [0.75030114, 1.24969886] |
| 真实平均rho范围，kg/m³ | [0.75040140, 1.24959860] |
| 点rho相对平均rho最大差 | 0.0133616% |
| 求积面积相对真实面积最大差 | **0.0133634%** |
| 原始求积面积总和 | **1.0000131689534162 m²** |
| 真实总面积 | 1 m² |

求积面积只保留为m/(rho_point*b)的权重解释。没有对其总和作归一化，也没有把它称为已经识别出的真实物质面积。面积相对偏差较小不能单独证明对算子误差的贡献较小，因为它还与非局部系数、状态差和消去的奇偶项共同作用。

## 4. 同一质量平均状态上的结果

统一用完整局部PDE变化率范数作为分母：norm²=sum m*r_full²。三项分别为

| 诊断 | 相对质量加权变化率范数 |
|---|---:|
| 候选与完整PDE的差 | **1.620938%** |
| 省略漂移的参考与完整PDE的差 | **10.817822%** |
| 候选与仅Laplacian诊断的距离 | **10.555657%** |

后两行是参考定义检查，没有运行一个删掉漂移的新候选。该状态中候选明显更接近完整变密度目标，不能由此宣称任意数据上的漂移都已准确恢复。原均匀rho档1.494108%的误差仅作背景；两个rho问题不同，不据此作方法改进百分比或误差分摊。

本轮最大绝对变化率误差0.00567323/s。4096节点、376832条非零共享边，保持连通；原始Q最小非对角为零。常数残差2.79e−14/s，质量残差1.08e−17 kg/s，加权对称性残差1.08e−19；独立共享图组装差1.91e−17，边通量散射与实际稀疏作用差2.64e−14/s。峰值RSS约135.3 MiB。

![光滑变密度状态、完整PDE与原始权重偏差](../../runs/reconnect-kernel2d-variable-density/variable-density.png)

## 5. 光滑连续极限为何应产生密度漂移

这部分是光滑条件下的理论解释，不作为候选所需的梯度输入，也不能直接跨理想激波使用。设Gamma=rho D，正且足够光滑，在连续求积极限下考虑

\[
\mathcal L_h c(x)=\frac{2}{\rho(x)}\int
H(\Gamma(x),\Gamma(x+r))B_h(|r|)[c(x+r)-c(x)]\,dr.
\]

调和平均在相同状态附近展开为

\[
H(\Gamma,\Gamma+\delta)=\Gamma+\delta/2+O(\delta^2).
\]

由径向核的奇矩抵消、int B_h r_alpha r_beta dr=delta_alpha,beta，二阶项给出

\[
\mathcal L_hc=\frac{\Gamma}{\rho}\Delta c+
\frac{\nabla\Gamma}{\rho}\cdot\nabla c+O(h^2)
=D\Delta c+D\nabla\log\rho\cdot\nabla c+O(h^2).
\]

O(h²)需要相应高阶光滑性、rho下界和无支撑截断；不含离散求积或有限物质平均闭合误差。因此原公式在这个光滑连续极限中具有正确的漂移主项，但当前1.6209%的有限h离散误差仍没有被唯一归因。

## 6. 检查、证据与限制

新增8项检查通过：解析/独立张量积分；复步物理通量散度；真实面通量余额；epsilon=0退化到原均匀控制；漂移重要性及守恒盲区；非负共享图和探针独立性；点密度求积权重不归一化；固定身份、oracle沿轨迹及状态非点值。此前244项，加8项累计记录252项；未重跑全部历史检查，不是旧246项恢复。

数据位于 `runs/reconnect-kernel2d-variable-density/`，包含预定配置、点输入、原始共享边/CSR算子、解析与Gauss8/16参考、漂移分项、权重差及原始误差。核验782个原始文件、15个历史附件及27份已有正式科学清单不变；4份更早诊断清单的既有路径差异单独保留。原始README/METHOD/CODEX_TASK、旧报告和旧运行结果未修改。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/test_smooth_density_probe.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/ltn-reference/bin/python research/reconnect/kernel2d_variable_density.py --output /workspace/kernel2d-variable-density-reproduction
```

只有一个光滑变密度周期状态；未建立变密度收敛，没有激波/接触、固壁、非均匀点云、密度重构/ledger、非平面或DNS验收。fluid ORC继续关闭。未生成PDF或版本，未提交、推送或发布。

## 7. 下一步只补同状态的连续核参照

不再新增N/h/rho档位或修改电导。保持当前保存的输入与算子，计算连续谐和系数核对真实物理标量的质量平均作用，并计算将当前单元平均状态与m/rho_point权重延拓后对应的连续求积作用。利用rho仅依赖x、标量为y余弦的分离结构，按64个独立x位置积分，不建立新大图。

这样可将当前误差分成：物理连续核的有限宽偏差、质量平均状态/点密度求积闭合差、最后的节点求积差。延拓公式必须由本轮已知解析输入导出，不拟合输出或修改候选。参考采用两组预定极坐标/单元积分（径向32、角向64、单元x向8；径向64、角向128、单元x向16），先核对积分差是否足以区分这三项，再决定后续路线。仍不运行新瞬态或ORC。
