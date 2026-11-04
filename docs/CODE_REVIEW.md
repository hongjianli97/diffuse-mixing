# 代码审查与迁移决定

本次递归检查14个外层 ZIP、37个含嵌套归档、1709条文件记录；699条 Python 记录去重后为272个不同源码，另有1个不同 notebook（8个代码单元）。全部 Python 完成 AST 语法解析及依赖/路径/危险回退模式扫描，未发现语法错误；notebook 普通 Python 单元解析通过。重点逐段审阅质量账本、扩散装配、掺混指标、时间推进、算例输入和测试入口。其余绘图、声学、压力等脚本做静态清点，不宣称逐个验证算法正确。

## 版本选择

reference_baseline/ 是从 LHD_v3.2_代码论文报告.zip 恢复的参考目录，保留原代码内容。v3.2 新增 ledger_v4、ledger3d、ns1d、t4split 等，但 netdiff.py 等大量核心模块与 v2.1 相同。“版本号更高”不代表扩散实现已升级。

prototypes/network_mixing/ 来自 LTN.zip 内网络掺混 v2；prototypes/optimization/ 来自 optimization_prototypes.zip；prototypes/interface_stretch/ 来自界面拉伸修正包。它们保留相对文件结构，但部分仍依赖历史外部路径和缺失数据，因此不是一键运行目录。

legacy_sources/ 保存去重源码、notebook、文本、LaTeX及历史 JSON/CSV 结果；相同文件以 audit/file_inventory.json 的 saved_as 查找，不能直接把某个 Axxx 目录当完整 Python 项目。原始 ZIP、PDF、图片、字节码和 xlsx 不重复装入代码交接包；成员、大小、哈希仍在清单中。原始归档无 npz/npy/pkl/h5 数据文件，因此多处 DNS 和 RM 运行依赖尚未满足。历史 JSON 结果不是本次重跑结果。

## 核心问题与优先级

|优先级|定位|观察与后果|迁移要求|
|---|---|---|---|
|P0|reference_baseline/lhd/netdiff.py: run/run2|M<=0 被替换为1，可能掩盖质量缺失|拒绝非法质量；有合法质量的孤立点保留零交换|
|P0|同上 stiffness|P1 正 Γ 可半正定但有正非对角；不能保证 Markov 核|保留 PDE 比较后端；ORC 前强制检查 Q 符号|
|P0|同上 stiffness_edge|调和导通率乘裁剪 cotan 和，与原始 P1 不等价|改名 experimental，记录裁剪及误差；不能把代数正性当连续一致性|
|P0|prototypes/network_mixing/netmix.py: Net.__init__|零质量用中位数替换；头部声称 exactly P1/material discretisation 过强|禁止质量补值；改为研究原型说明|
|P0|reference_baseline/lhd/bcom.py: sph_diffusion|只保留 query_ball 的 i<j，变 h 下可能漏掉合法无向边|以对称支撑构造边集，加入置换不变性测试|
|P0|reference_baseline/lhd/ledger_v4.py|冲击确认、拟合和分段修正仍依赖局部 F|不得成为无 C/无梯度主线的强制依赖；只作对照|
|P0|prototypes/optimization/sbcore.py: Prep.diffuse|cg 返回 info 未检查|任何非收敛必须显式失败，输出 residual|
|P0|ORC|全部 Python 及 notebook 未检索到 Ollivier/Wasserstein/LP 实现；Ricci 命中的是 Forman|按 METHOD 第8节新建并测试；旧 ORC 报告不算可执行实现|
|P1|netdiff.run2: theta 分支|固定 M 时常规 theta；volume 下 M 随时间变，右端 Ml*c 与 run 的含义不能简单视为同一对照 PDE|首版只启用固定质量；volume 实验独立标明模型|
|P1|netdiff.run2: state|步内位置线性、密度对数插值跨激波可能展宽/错位|记录时间模型；理想跳变事件与分辨层采样分开|
|P1|netmix.diffuse/Lbar|平均 L 后一步隐式推进/全窗平均算子不能替代精确时序传播|保存逐步算子，P按时间左乘；平均仅作对照|
|P1|netmix.mixed_fraction|分母为初始标签质量，结果不必≤1|主指标换为质量方差混合度；旧指标改名|
|P1|netmix.cut|剥皮与裁剪后 w*length 不能无条件称精确物质界面长|视为网络接触几何代理，与独立 marker 对照|
|P1|netmix.far_contacts|强各向异性形变下参考远邻判据不保证只识别折叠|用仿射剪切/压缩负对照，不当作严格折叠定理|
|P1|ledger_v4/stream/bcom|有 NaN 填零、保持旧值、单侧修补和阈值|输出有效性掩码与修补比例，不当作无偏真实密度|
|P1|scripts/v5_diffusion.py|exec 从其他源码切字符串；无效密度中位数补值|重构显式函数导入；保留误差分解思路，取消隐式修补|
|P1|多处 /home/claude/w 路径|依赖旧运行环境；若直接 import 可失败或执行长任务|config/CLI 指定数据路径；脚本主逻辑放 main guard|
|P2|thermo_from_ledger|由压缩重建温度需要气体/熵跃迁等假设|不放基础运动学主线，提供外部本构输入|
|P2|stretch_est|含 F 链与层片模型|可选光滑区解释，不作为激波扩散主求解器|

## 本次实际运行的验证

执行 audit/run_numeric_audit.py，结果 audit/numeric_audit.json。固定随机种子2048、45点二维 Delaunay、正但强非均匀 Γ：
- element P1 出现44个正非对角条目；这足以否定普遍 Markov 保证，但本次随机 c 一步没有证明发生越界。
- edge_harm 正非对角0，两个后端均保持对称和行和零到浮点误差；一步相对质量漂移约3–4×10⁻17，能量下降。
- Γ全零时 edge_harm 返回零矩阵，但除零警告仍存在，应显式零分支。
- 两点、h=(0.1,1)、距离1的核支撑反例：平均核支撑1.1应包含边，旧筛选返回空边集。需修复，尚未修改历史源码。

尝试运行 tests/test_identities_exact.py，当前两种 Python 入口均缺 sympy，未运行成功；不能标记旧符号测试通过。没有运行完整 DNS、RM 或 ORC 验证，也没有验证旧报告的百分比数值。AST通过只代表语法可解析，不代表导入/计算正确。

## 依赖和数据

主要第三方：numpy、scipy、matplotlib；符号测试 sympy；部分图像/轮廓处理 scikit-image。标准库和内部模块见 audit/import_inventory.json。历史源码裸 import lhdx/lhd 与 lhd.py 命名冲突，依赖从 reference_baseline 根目录运行并将 lhd 加入 sys.path；新包改成显式相对导入。

典型缺失文件：dns_sb.pkl、dns_sv.pkl、sb_net_m.pkl、sb_lite_m.npz；其他具体外部加载语句见 audit/static_findings.json。不要凭文件名伪造数据。旧 pickle 数据应仅加载可信输入；新格式用 npz（allow_pickle=False）/JSON 或有清晰 schema 的 HDF5。

## 复用顺序

先复用 stiffness 的数学参考、cases 中的解析算例思想、轨迹与边几何工具。随后移植 marker 参照与密度账本为可选适配器。网络质量、数据契约、生成元、时间传播、ORC、诊断均按新规范组织。不要整体 import 绘图/运行脚本，也不要直接把旧项目目录合并造成同名函数覆盖。
