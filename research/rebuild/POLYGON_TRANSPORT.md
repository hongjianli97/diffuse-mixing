# 第三阶段：二维固定物质单元的面积与共享边输运

日期：2026-10-31。Stage 3 **incomplete**，fluid ORC **关闭**。本轮完成二维几何记录及独立质量输运审计，**没有运行二维标量扩散求解或DNS验收**。

64个固定物质四边形、99个顶点、162条唯一边中，94条内部共享边严格反向使用。中点边界扫掠重构面积增量的残差约 `3e-17`，但三档采样的实际流体质量偏差仍约5.48%、1.38%、0.347%。几何守恒和流体物质性继续作为不同问题检查。

## 1. 固定输入，不动态重连

实现 [MaterialPolygonHistory](material_polygon_history.py)，显式给定顶点ID、单元ID、逆时针顶点连接、固定m、c0、SI单位、单位厚度和封闭边界。细节见 [二维输入约定](MATERIAL_POLYGON_INPUT.md)。保存/加载会复核派生共享边及方向。

本轮复用原Becker32列物质质量和初场，将原1 m²横截面分成两个0.5m宽、1m厚条带，得到32×2个单元。每个子质量为原质量的一半，原初始质量平均浓度在两条带复制，合计质量与原数据逐值核对无差异。原一维结果及定义不改变。

三档采样的身份/连接/质量摘要相同：

`e21625d8e0daec2d9ac53b1bc3de3ba75ac3747bd3a41e56afbab2046da5a950`

## 2. 一个可独立积分的二维连续性基准

以原Becker物质运动 `xi=X_B(a,t)`、`eta=b` 为基础，定义给定运动

\[
Y(a,b,t)=R(\theta(t))
\begin{pmatrix}1&k(t)\\0&1\end{pmatrix}
\begin{pmatrix}X_B(a,t)\\b\end{pmatrix}+s(t),
\]
\[
\theta(t)=\pi t/(3T),\qquad k(t)=0.6\sin(\pi t/T),\qquad
s(t)=(0.1t/T,-0.05t/T),\quad T=0.1.
\]

旋转和剪切的面积因子为1，密度从原一维场推前。用该映射的时间导数给出独立流体速度，可直接满足连续性方程。它是一个**给定的二维运动学连续性基准**，不是无外力二维Navier–Stokes激波或DNS解。给定解析映射仅用于产生和验证基准；几何接口不拟合F/C，也不需要该矩阵作为输入。

本例中原矩形的各条边在真实运动下仍为直线，真单元为平行四边形，其精确面积为

\[
A_{ij}(t)=[X_B(a_{i+1},t)-X_B(a_i,t)]\,\Delta b_j.
\]

单元实际质量可独立积分：先逆变换到xi、eta坐标，用原解析质量坐标对xi方向精确积分，再对eta方向作Gauss积分。另在最大质量偏差样本处，用二维密度张量积分交叉核对。

顶点在采样结点来自合成真物质轨迹；接口只接收序列化的坐标和连接，不查询oracle密度。真实几何仍由基准提供，不能称为一般节点点云的物质边界恢复。

## 3. 二维插值必须检查整个区间

顶点线性插值使有向面积及局部转角行列式都成为采样参数的二次多项式。接口计算区间最小值，拒绝中间非正面积或失去严格凸性的记录。

保留一个明确负对照：单位正方形的两帧坐标分别为p和−p，两端有向面积均为1，线性插值中点却全部塌缩，面积为0。记录被正确拒绝，原始位置及拒绝原因保存为 `rejected-half-turn.npz` 和结果JSON。没有用面积绝对值、删单元或修补通过。

当前接口检查局部凸四边形和共享边关系，不声称一般全局无重叠证明。本例真映射可逆；插值网格保留公共条带方向，严格正面积使各条带有序，因此可单独验证其不重叠。这个特殊结构不推广为任意非结构网格的保证。

## 4. 面积增量的正确共享边规则

对全局有向边ab，定义几何扫掠通量

\[
q_e(t)=\frac{w_a+w_b}{2}\cdot
       (y_b-y_a,-(x_b-x_a)).
\]

单元按逆时针边界使用符号σ；内部边只计算一次，两个相邻单元符号相反。对于线性顶点运动，q_e随时间线性，故

\[
\Delta A_i=\Delta t\sum_e\sigma_{ie}q_e(t_{mid}).
\]

中点规则对这个几何模型是精确积分，不是改变旧FV标量时间方法。左端几何规则作为失败负对照保留：

| 顶点采样区间数 | 正确中点扫掠的最大面积增量残差 | 左端几何负对照残差 |
|---:|---:|---:|
| 8 | 3.43475e-17 | 1.47126e-3 |
| 16 | 3.05311e-17 | 3.73058e-4 |
| 32 | 3.08130e-17 | 9.35951e-5 |

内部边方向和为整数零；全部单元面积增量与外边界扫掠量的差不超过 `1.69e-16`。这些量只验证几何模型，不证明帧间真实流体质量恰好等于m。

## 5. 实际流体质量与Reynolds输运独立验证

采样区间数仍固定为8、16、32。每档使用513个审计时刻，另检查其257点子集；最大质量偏差在两组离散时刻上相同，但这不构成连续时间严格上界。

| 区间数 | 最大采样面积偏差 | 最大采样实际流体质量偏差 |
|---:|---:|---:|
| 8 | 1.77147% | 5.47792% |
| 16 | 0.459505% | 1.37721% |
| 32 | 0.118584% | 0.346740% |

面积成对观察阶约1.947、1.954，质量偏差约1.992、1.990。它们支持当前光滑基准的接近二阶采样趋势，不是任意二维流动的收敛定理。采样结点的质量偏差仅约 `2.63e-14`；只核对记录结点会遗漏帧间误差。

独立质量审计采用16/32点Gauss控制，最大相对质量差不超过 `8.78e-15`；直接二维密度积分与质量原函数积分在被选最坏样本处相差不超过 `6.94e-18`。这些是数值交叉检查，不是严格舍入误差定理。

再对唯一共享边计算实际相对质量通量

\[
F_e=\int_e\rho(u-w)\cdot n_e\,ds,\qquad
\dot M_i=-\sum_e\sigma_{ie}F_e.
\]

边上流体密度和速度来自独立解析基准，网格速度来自记录线性插值。这个F_e是有向物理质量通量，不是扩散导通率或拟合参数。

在每个采样区间中点，用该边通量计算的质量变化率，与独立质量积分的中心时间差分核对：

| 区间数 | 最大Reynolds变化率残差 | 中点样本的最大质量变化率绝对值 |
|---:|---:|---:|
| 8 | 1.41172e-9 | 5.28752e-2 |
| 16 | 1.42329e-9 | 1.33891e-2 |
| 32 | 1.37475e-9 | 3.36789e-3 |

时间差分增量预先固定为1e-6；通量16/32点积分差不超过 `1.77e-15`。全域质量变化率与外边界通量差不超过 `2.26e-17`。质量变化率最大值只指这些中点样本，不声称是整个区间的峰值。

这些检查共同确认：几何共享边抵消可以达到机器精度，帧间真实流体质量误差仍需作为独立的物质性误差保留。没有向单元补质量或将密度重归一化。

![二维网格、面积守恒与真实质量审计](../../runs/rebuild-stage3-polygon-transport/polygon-transport.png)

## 6. 新增检查、复现及下一步

本轮新增10项检查通过：面积与独立伸缩算例、中间塌缩拒绝、有效90度旋转的二次最小面积、共享边反向、非法连接/身份/单位拒绝、只读持久化、时间/质量拒绝、真二维质量/面积、Reynolds与独立积分、质量尺度与密度。历轮记录为 `127+10=137` 项，不是本轮全部重跑或旧246项恢复；这些检查不被计作二维扩散验收。

新增源码：`material_polygon_history.py, becker_polygon_benchmark.py, polygon_transport_study.py, test_polygon_transport.py, plot_polygon_transport.py`。接口说明：[MATERIAL_POLYGON_INPUT.md](MATERIAL_POLYGON_INPUT.md)。数据入口：`runs/rebuild-stage3-polygon-transport/`，保留三份可重载网格、513帧原始审计、共享边方向、质量及面积通量、Reynolds检查、拒绝记录和配置哈希。

从仓库根目录运行，使用现有Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0；输出目录需不存在：

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/ltn-reference/bin/python research/rebuild/test_polygon_transport.py
/workspace/.venvs/ltn-reference/bin/python research/rebuild/polygon_transport_study.py runs/rebuild-stage3-face-history --output /tmp/ltn-polygon-transport-new
MPLCONFIGDIR=/workspace/.cache/ltnmix/matplotlib XDG_CACHE_HOME=/workspace/.cache/ltnmix/xdg /workspace/.venvs/ltn-reference/bin/python research/rebuild/plot_polygon_transport.py /tmp/ltn-polygon-transport-new --output /tmp/ltn-polygon-transport-new/polygon-transport.png
```

下一步是在这类非正交单元上检验线性标量的面通量一致性，再决定二维扩散离散。面积账本和质量输运核对不能替代扩散通量检验，一维中心阻力也不自动适用于二维非正交面。Stage 3全面物理验收、二维非平面DNS和fluid ORC继续关闭或待验收。

原始782文件、15附件和全部旧科学结果继续核验；不修改原README/METHOD/CODEX_TASK或导入附件，不生成PDF、新软件版本、推送或发布，也不改环境配置。
