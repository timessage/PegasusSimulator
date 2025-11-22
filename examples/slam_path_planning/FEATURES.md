# SLAM建图与路径规划项目功能特性 / SLAM Features

## 功能概览 / Feature Overview

本项目实现了完整的2D和3D SLAM系统，用于无人机的自主导航。

This project implements a complete 2D and 3D SLAM system for autonomous drone navigation.

---

## 核心算法 / Core Algorithms

### 1. 概率建图 / Probabilistic Mapping

**算法**: 贝叶斯对数几率更新 (Bayesian Log-Odds Update)

```
log_odds_new = log_odds_old + Δlog_odds
probability = 1 - 1/(1 + exp(log_odds))
```

**优势 / Advantages**:
- 高效的数值稳定性
- 避免概率边界问题 (0和1)
- 支持增量更新

### 2. 射线追踪 / Ray Tracing

**算法**: 3D Bresenham算法

用于从激光雷达数据更新占据网格:
- 射线路径上的单元 → 自由空间
- 射线终点 → 障碍物

**复杂度 / Complexity**: O(d) where d = 射线长度 (ray length)

### 3. 路径规划 / Path Planning

**算法**: A* 搜索算法

**2D版本**:
- 8连通 (允许对角线移动)
- 启发式: 欧几里得距离

**3D版本**:
- 26连通 (所有相邻体素)
- 启发式: 3D欧几里得距离
- 优选高度: 能耗优化

**复杂度 / Complexity**: O(b^d) where:
- b = 分支因子 (8 or 26)
- d = 解的深度

### 4. 路径平滑 / Path Smoothing

**算法**: 移动平均滤波器

```python
smoothed_point = average(points[i-w:i+w])
```

保持起点和终点固定，平滑中间路径点。

---

## 技术规格 / Technical Specifications

### 2D系统 / 2D System

| 特性 | 规格 |
|------|------|
| 网格类型 | 2D占据栅格 |
| 分辨率 | 0.1-0.5米可配置 |
| 最大地图尺寸 | 100m × 100m |
| 路径规划 | 8连通A* |
| 更新速度 | >100Hz |

### 3D系统 / 3D System

| 特性 | 规格 |
|------|------|
| 网格类型 | 3D体素网格 |
| 点云容量 | 最多50万点 |
| 分辨率 | 0.2-0.5米可配置 |
| 最大地图尺寸 | 100m × 100m × 20m |
| 路径规划 | 26连通3D A* |
| 体素过滤 | 自动去重 |
| 导出格式 | NPZ, PLY |

---

## 性能特性 / Performance Features

### 内存优化 / Memory Optimization

1. **体素过滤**: 同一体素只保留一个点
2. **增量更新**: 只更新受影响的区域
3. **稀疏表示**: 使用字典存储已知体素

### 计算优化 / Computational Optimization

1. **A*优化**:
   - 优先队列实现
   - 启发式函数优化
   - 早期终止

2. **射线追踪优化**:
   - 整数算法 (Bresenham)
   - 避免浮点运算
   - 边界检查优化

3. **并行化潜力**:
   - 射线追踪可并行
   - 地图更新可分区

---

## 可视化功能 / Visualization Features

### 2D可视化

- 占据概率热图
- 路径叠加显示
- 机器人和目标标记
- 统计信息文本框

### 3D可视化

- **多视角**:
  - 等轴测视图
  - 前视图 (YZ平面)
  - 侧视图 (XZ平面)
  - 俯视图 (XY平面)

- **点云渲染**:
  - 高度着色
  - 自定义颜色
  - 透明度控制

- **组合视图**:
  - 3D点云 + 2D投影
  - 路径可视化
  - 统计信息

- **导出功能**:
  - PNG高分辨率图像
  - PLY点云格式
  - 动画帧序列

---

## 控制策略 / Control Strategies

### 位置控制 / Position Control

比例控制器:
```
v_desired = K_p * (target - current)
v_clamped = clamp(v_desired, -v_max, v_max)
```

### 高度控制 / Altitude Control

独立垂直速度控制:
```
v_z = K_z * (target_z - current_z)
v_z_clamped = clamp(v_z, -v_z_max, v_z_max)
```

### 偏航控制 / Yaw Control

角速度比例控制:
```
yaw_error = normalize_angle(target_yaw - current_yaw)
yaw_rate = K_yaw * yaw_error
```

---

## 安全特性 / Safety Features

1. **障碍物膨胀**:
   - 可配置安全半径
   - 3D膨胀 (所有方向)

2. **速度限制**:
   - 最大水平速度
   - 最大垂直速度
   - 最大偏航角速度

3. **边界检查**:
   - 地图边界验证
   - 高度限制
   - 无效点过滤

4. **动态重规划**:
   - 检测路径障碍物
   - 自动重新规划
   - 平滑路径切换

---

## 扩展性 / Extensibility

### 易于扩展的功能

1. **传感器融合**:
   - 添加相机数据
   - IMU数据融合
   - GPS定位

2. **高级规划**:
   - RRT/RRT* 算法
   - 动态规划
   - 强化学习

3. **多机协同**:
   - 分布式建图
   - 协同路径规划
   - 地图合并

4. **回环检测**:
   - 地图一致性
   - 漂移校正
   - 全局优化

---

## 性能基准 / Performance Benchmarks

### 典型场景性能

| 操作 | 2D | 3D | 单位 |
|------|----|----|------|
| 地图更新 | 5 | 15 | ms |
| 路径规划 | 50 | 200 | ms |
| 路径跟随 | 1 | 1 | ms |
| 可视化 | 100 | 500 | ms |

*注: 性能取决于地图大小和复杂度*

### 可扩展性

- **地图大小**: 线性复杂度
- **点云点数**: 对数复杂度 (体素过滤)
- **路径长度**: 对数复杂度 (A*)

---

## 应用场景 / Applications

### 适用场景

✅ 室内导航
✅ 室外探索
✅ 仓库巡检
✅ 搜索救援
✅ 环境监测
✅ 3D建模

### 限制

⚠ 需要良好的激光雷达覆盖
⚠ 动态环境需要更高更新频率
⚠ 大型环境需要更多内存

---

## 未来改进方向 / Future Improvements

### 短期 (Short-term)
- [ ] 增加更多传感器支持
- [ ] 优化大规模地图内存使用
- [ ] 添加实时性能监控

### 中期 (Mid-term)
- [ ] 实现回环检测
- [ ] 添加语义分割
- [ ] 支持多机器人协同

### 长期 (Long-term)
- [ ] 深度学习路径规划
- [ ] 端到端学习系统
- [ ] 完全自主探索

---

## 参考文献 / References

1. **SLAM算法**:
   - Thrun, S., et al. (2005). "Probabilistic Robotics"
   - Grisetti, G., et al. (2010). "A Tutorial on Graph-Based SLAM"

2. **路径规划**:
   - Hart, P. E., et al. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
   - LaValle, S. M. (2006). "Planning Algorithms"

3. **3D建图**:
   - Hornung, A., et al. (2013). "OctoMap: An Efficient Probabilistic 3D Mapping Framework"
   - Rusu, R. B., et al. (2011). "3D is here: Point Cloud Library (PCL)"

---

## 许可证 / License

BSD-3-Clause License
Copyright (c) 2024. All rights reserved.
