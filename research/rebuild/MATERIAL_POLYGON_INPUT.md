# 二维固定物质四边形记录接口

实现：[material_polygon_history.py](material_polygon_history.py)。这是固定连接、单位厚度、封闭物质边界的二维研究接口。当前只支持严格凸的逆时针四边形组成的连通、盘状边关系，不处理重剖分、曲边或一般全局无重叠证明。

## 输入与身份

| 字段 | 内容 |
|---|---|
| time | S≥2个有限严格递增采样时刻 |
| positions | S×V×2顶点坐标，列身份全时段固定 |
| vertex_ids | V个唯一字符串或整数ID |
| cell_ids | C个唯一单元ID |
| cell_vertices | C×4个顶点ID，逐单元逆时针有向连接 |
| mass, c0 | 每个单元固定正质量、初始质量平均浓度 |
| units | time=s、position=m、mass=kg、thickness=m |
| thickness | 当前为1 m |
| boundary_mode | 当前为material_no_flux |
| metadata | geometry_kind=material_quadrilateral_cells及来源 |
| frame_vertex_ids | 可选逐帧ID，提供时必须与全局列身份一致 |

连接必须引用已知顶点，每单元四个顶点不同。重复单元、未使用顶点、共享边超过两个单元、共享边方向不相反、断连和不符合当前边关系限制的复形均拒绝。数值数组复制后只读；不会根据坐标重新排序顶点或重映射质量。

共享边由固定顶点ID对派生一次，保存全局edge_ids、cell_edge_ids及逐单元方向。加载NPZ时重新推导并核对这些字段；不允许逐帧各自重建共享边。identity_digest包含身份、连接、质量、初场和单位，不因时间采样密度变化而改变。

## 区间有效性与面积

区间内顶点作共享线性插值，顶点速度为该区间的常数。与一维不同，二维有向面积随插值参数为二次函数。接口检查每个区间上的有向面积和四个局部转角行列式的二次多项式最小值，拒绝中间退化或失去严格凸性的单元；不对面积取绝对值或裁剪。

这些是浮点数值下的局部有效性检查。输入生产者仍需保证全局嵌入和真实物质边界含义。局部正面积、盘状边关系和metadata声明不能自行证明一般全局无重叠或真实流体物质性。采样顶点也不自动给出任意曲边物质面的精确形状。

## 共享边几何输运

设某条全局有向边端点为a、b，边切向量为e，顶点速度为wa、wb。全局方向的面积通量为

\[
q_e=\frac{w_a+w_b}{2}\cdot(e_y,-e_x).
\]

单元按自身逆时针边界方向使用符号σ，面积变化率为 `Σσq_e`。相邻单元复用同一个q_e并取相反符号，所以内部边逐条抵消。

线性顶点运动使e随时间线性变化，故q_e也线性。区间面积增量精确对应

\[
A_i^{s+1}-A_i^s=\Delta t_s\sum_e\sigma_{ie}q_e(t_{s+1/2}).
\]

这里的中点是几何扫掠积分规则，不是对原标量扩散时间格式的替换。接口返回面积、m/面积密度、顶点速度、唯一边通量和单元面积变化率；这些有向几何量不是物理扩散导通率。

## 持久化与示例

save拒绝覆盖；load禁用pickle，并复核身份、局部区间有效性及共享边方向。查询只允许采样范围内的标量或一维时间数组；不外推，内结点速度采用右区间、末端采用左区间。

从项目根目录读取本轮样例：

```bash
PYTHONPATH=research/rebuild /workspace/.venvs/ltn-reference/bin/python - <<'PY'
from material_polygon_history import MaterialPolygonHistory
mesh=MaterialPolygonHistory.load('runs/rebuild-stage3-polygon-transport/polygon-record-32.npz')
geometry=mesh.evaluate([0.,0.05,0.1])
print(mesh.identity_digest)
print(geometry['area'].shape, geometry['edge_area_flux'].shape)
print(mesh.geometric_conservation())
PY
```

接口只消费顶点位置、连接、固定质量和单位，不消费oracle密度或拟合F/C。独立流体质量验证与本轮限制见 [POLYGON_TRANSPORT.md](POLYGON_TRANSPORT.md)。
