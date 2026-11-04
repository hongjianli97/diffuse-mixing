# 精确解验证（第二批）与场图
suite.py <linear|lamb|tg|vortex|acoustic|sod|lamella>：第一批精确解；运行后同时保存 fields_<case>.npz 供作图
suite2.py <stokes|igw|conv|noise>：
  stokes：Stokes 第一问题（壁面、粘性、非定常精确解），墙作自由边界 vs 墙外延拓（像轨迹 = 全平面精确解）
  igw：线性内重力波（Boussinesq），强斜压力矩，N² 由位置反演
  conv：Taylor–Green 三档示踪点间距的收敛
  noise：Taylor–Green 位置噪声敏感性
viz.py：图 16（涡量/压力/熵/耗散场图）、图 17（声学/掺混/重力波/壁面）、图 18（收敛与噪声）
注意：解析轨迹若用细时间网格积分，细网格必须与帧时刻对齐，否则二阶差分加速度会被时间错位污染（Stokes 算例的第一次运行即为此错误）。
