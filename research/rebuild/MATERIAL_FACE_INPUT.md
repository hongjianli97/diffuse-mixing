# 一维固定物质面输入约定

新增研究接口：[material_face_history.py](material_face_history.py)。适用范围为一维、N≥2个单元、单位横截面积、固定连接和身份、封闭物质边界。它不是旧ltnmix数据接口的恢复，也不代表物质面来源已经通过物理验收。

## 输入字段

| 字段 | 形状/内容 | 必须满足 |
|---|---|---|
| time | S个采样时间 | S≥2、有限、严格递增 |
| positions | S×(N+1)面位置 | 有限；每帧按既定物质身份严格递增 |
| face_ids | N+1个字符串或整数ID | 唯一、固定列含义，不从位置重新排序 |
| cell_ids | N个字符串或整数ID | 唯一、全时段固定 |
| cell_faces | N×2个面ID | 依次等于face_ids中的相邻有向面；共享面仅有一个身份 |
| mass | N个质量 | 有限、正、全时段固定 |
| c0 | N个初始质量分数平均 | 有限、在[0,1]内；不修补 |
| units | 明确SI字典 | time=s，position=m，mass=kg，cross_section_area=m² |
| cross_section_area | 标量 | 当前只接受1 m² |
| boundary_mode | 字符串 | 当前只接受material_no_flux |
| metadata | 来源信息 | geometry_kind必须声明material_faces |
| frame_face_ids | 可选S×(N+1) | 若提供，逐帧必须与face_ids一致 |

构造时复制数值数组并置为只读，不共享调用方随后修改的数据。改变采样密度不改变质量、初场、身份和连接；identity_digest单独记录这些不变输入，不包含采样时间或位置值。元数据中的物质面声明不能从坐标和编号自行证明。

缺失、重排、交叉、重复身份、非正体积、非有限值、单位不符均拒绝。没有自动排序、质量重映射、填零或保留旧值。当前不支持开边界、周期闭合、出生/丢失单元或拓扑变化。

## 插值和几何守恒

在相邻采样时刻之间，对所有共享面用同一个线性时间权重：

\[
\widehat b_f(t)=(1-\theta)b_f^s+\theta b_f^{s+1},\qquad
w_f=(b_f^{s+1}-b_f^s)/(t_{s+1}-t_s).
\]

单元体积同样是两端正体积的凸组合，因此保持正性。单位横截面下，逐区间满足

\[
V_i^{s+1}-V_i^s=\Delta t_s(w_{i+1}-w_i).
\]

接口返回插值位置、分段常值面速度、体积、密度m/V和所用采样区间。中间结点采用右区间速度，最终结点采用左区间速度；位置在全部输入结点精确重现。查询仅允许标量或一维时间数组，不外推。

**上述几何守恒不是流体物质性证明。** 即使位置在输入结点是真实物质面，帧间线段也可能偏离真实面轨迹。其实际流体质量及相对流体速度的质量通量应另行验证；参见 [采样验证](FACE_SAMPLING.md)。接口不将w当作已知真实流体速度。

## 持久化与FV使用

NPZ记录包含全部字段、明确的units_json、metadata_json及逐帧ID副本；加载禁用pickle。save拒绝覆盖已有文件。构造与加载执行相同的验证，保存/加载后identity_digest和数组值保持一致。

从项目根目录可读取本轮真实样例：

```bash
PYTHONPATH=research/rebuild /workspace/.venvs/ltn-reference/bin/python - <<'PY'
from material_face_history import MaterialFaceHistory
history=MaterialFaceHistory.load('runs/rebuild-stage3-face-history/face-record-16.npz')
geometry=history.evaluate([0.,0.05,0.1])
g=history.conductance([0.,0.05,0.1],D=0.02)
print(history.identity_digest)
print(geometry['volume'].shape, g.shape)
print(history.geometric_conservation())
PY
```

conductance使用同一片常值平均密度中心阻力公式。接口不消费oracle密度、F、C或速度梯度。它需要物质边界的位置历史，不能用内部点轨迹的动态双面替代这一输入。
