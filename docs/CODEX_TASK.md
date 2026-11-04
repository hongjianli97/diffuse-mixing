# 交给 Codex 的实施任务

请先读 README.md、docs/METHOD.md、docs/CODE_REVIEW.md；以 METHOD 为新实现规范。交付的是新的研究代码包，不修改 reference_baseline 和 legacy_sources 原始文件。不得将历史 JSON 结果作为新实验结果。

## 第一阶段：可验证的图层核心

建立 src/ltnmix/，模块 data.py、mass.py、geometry.py、conductance.py、operators.py、propagator.py、mixing.py、orc.py、diagnostics.py、cli.py。先实现 supplied_graph 与小图，不提前投入大 DNS。

建议接口（签名为设计契约，不是现有代码）：
- TrajectoryData(time[S], positions[S,N,d], ids[N], mass[N], rho[S,N]|None, c0[N], boundary, metadata)。validate() 验证维度、单位、正质量和全时段固定身份。
- ReferenceGraph(edges[E0,2], lengths[E0], metric_metadata)。reference_distances(...) 输出参考最短路；不跨壁。
- ExchangeGraph(edges[E,2], g[E], diagnostics)。无向边 i<j，禁止重复；zero g 允许。
- build_laplacian(n, graph) -> CSR L；generator(m,L) -> Q；validate_operator 检查行和、质量、符号和对称性。
- step(c,m,L,dt) -> c_next, diagnostics；propagate(...) 保存时间顺序与实际能量收支。
- kernel_rows(indices, steps) -> P[indices,:]；用转置求解提取行，不能把作用于 e_i 的列误认为行。
- mixing_metrics(m,c,c0) 输出均值、E、I、H、χ（需要 L）；处理初态常量。
- wasserstein_lp(mu,nu,cost) 返回原始/对偶值、gap和边缘残差；orc(P_rows,d0,probe_edges) 返回曲率与状态。

先完成二节点不等质量、非对易三节点、D=0、常数标量、图重标号、参考尺度和不同 m 的测试。给出 CLI 最小演示和可复现 JSON。线性代数容差可设相对1e−10，但不能等同物理精度。

## 第二阶段：流体空间后端

接入 oracle rho；实现 p1_reference 和 positive_exchange 候选，保留后端差异。首先静止介质正弦扩散与无通量边界，再光滑运动。修复核邻居漏边，测试非均匀 h 与节点置换。所有实验保存网格/粒子尺度、边界、导通率量纲、ρ与D来源、裁剪/剥皮比例。

只有空间收敛和物理基准通过后，才启用 ORC 流场解释。p1_reference 若 Q 非对角有负值，明确拒绝 ORC；不得用 abs 或任意截断强行运行。

## 第三阶段：质量密度与激波

引入一维移动激波/接触间断 oracle 参考；点值跳变与有限区域平均分开。oracle、kernel、ledger 三种密度来源使用同一交换后端与物理 D 比较。ledger 是实验适配器，默认不启用 F 修正，不要擅自以旧 v4 替换无梯度路线。

平面激波上的特殊精确关系不外推到曲面非定常激波。跨冲击质量条件只适用于非零质量通量；接触面不能除以零推密度比。分别细化 h、Δt、物理层厚度δ。没有原始 DNS 数据时运行解析小算例，并清楚报告缺失项。

## 第四阶段：动态 ORC 与论文算例

小图精确 OT→抽样参考边→大图近似误差控制。对比 ordered 与 averaged 传播、固定参考距离与可选内禀距离、恒D与恒Γ。每条 ORC 结论同时配实际 c0 的质量混合度。最后接入双涡、blast、激波气泡/RM。

## 完成条件

提供 pyproject.toml、版本记录、CLI、数据 schema、最小数据生成器、tests、运行日志及图表脚本；CPU 小算例可在干净环境运行。每次提交说明哪些理论性质已验证、哪些物理一致性仍待验证。不得宣称全部已验证，不得把单个随机测试通过视作定理，不得悄悄更改 PDE、m、D、边界或度量使结果“更好”。

先执行第一阶段并交付可运行结果；其余按验收条件推进。研究假设不能在没有证据时升级为生产默认设置。
