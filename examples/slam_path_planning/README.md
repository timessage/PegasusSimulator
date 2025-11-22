# SLAM建图与路径规划项目 / SLAM Mapping and Path Planning Project

[中文](#中文文档) | [English](#english-documentation)

---

## 中文文档

### 项目简介

这是一个完整的SLAM（同步定位与建图）和路径规划项目，基于Pegasus Simulator开发。该项目展示了如何使用激光雷达传感器进行实时地图构建，并使用A*算法进行路径规划和自主导航。

### 主要功能

1. **实时SLAM建图**
   - 使用激光雷达数据进行2D占据栅格地图构建
   - 概率方法更新地图（使用对数几率）
   - 支持地图保存和加载

2. **A*路径规划**
   - 基于占据栅格的A*搜索算法
   - 支持对角线移动（8连通）
   - 障碍物膨胀以提供安全边界
   - 路径平滑算法

3. **自主导航**
   - 路径跟踪控制器
   - 实时避障
   - 多航点导航支持

### 项目结构

```
slam_path_planning/
├── occupancy_grid.py      # 占据栅格地图实现
├── path_planner.py        # A*路径规划算法
├── slam_controller.py     # SLAM控制器（整合建图和导航）
└── README.md             # 本文档

slam_path_planning_example.py  # 主程序示例
```

### 核心组件详解

#### 1. OccupancyGrid（占据栅格）

**文件**: `occupancy_grid.py`

占据栅格是SLAM系统的核心数据结构，用于表示环境地图。

**主要特性**：
- 使用对数几率表示占据概率
- 支持从激光雷达数据更新地图
- Bresenham射线追踪算法
- 概率和二值地图转换

**关键参数**：
```python
OccupancyGrid(
    width=50.0,        # 地图宽度（米）
    height=50.0,       # 地图高度（米）
    resolution=0.1,    # 栅格分辨率（米/格）
    origin=(0.0, 0.0)  # 地图原点坐标
)
```

**核心方法**：
- `update_from_lidar()`: 使用激光雷达数据更新地图
- `get_probability_grid()`: 获取概率地图
- `is_cell_free()`: 检查单元格是否空闲

#### 2. AStarPlanner（A*路径规划器）

**文件**: `path_planner.py`

实现了经典的A*搜索算法，用于在占据栅格中寻找最优路径。

**主要特性**：
- 支持4连通或8连通移动
- 欧几里得启发式函数
- 障碍物膨胀
- 路径平滑

**关键参数**：
```python
AStarPlanner(
    occupancy_grid,        # 占据栅格对象
    allow_diagonal=True    # 是否允许对角线移动
)
```

**核心方法**：
- `plan()`: 规划从起点到终点的路径
- `_astar_search()`: A*搜索实现
- `_inflate_obstacles()`: 障碍物膨胀

#### 3. SLAMController（SLAM控制器）

**文件**: `slam_controller.py`

整合SLAM建图、路径规划和运动控制的高级控制器。

**主要特性**：
- 实时地图更新
- 自动路径规划
- 路径跟踪控制
- 探索目标生成
- 地图保存

**关键参数**：
```python
SLAMController(
    map_width=50.0,              # 地图宽度
    map_height=50.0,             # 地图高度
    map_resolution=0.2,          # 地图分辨率
    map_origin=(-25.0, -25.0)   # 地图原点
)
```

**核心方法**：
- `update()`: 主更新循环
- `set_goal()`: 设置导航目标
- `update_map()`: 更新地图
- `save_map()`: 保存地图

### 使用方法

#### 前置要求

1. 安装Isaac Sim 5.1.0或更高版本
2. 安装Pegasus Simulator
3. Python 3.7+
4. 依赖包：numpy, scipy

#### 运行示例

```bash
# 确保已经设置了Isaac Sim环境变量
# 运行SLAM路径规划示例
isaac_run examples/slam_path_planning_example.py
```

#### 自定义配置

可以在 `slam_path_planning_example.py` 中修改以下参数：

**航点配置**：
```python
waypoints = [
    (8.0, 0.0),    # 第一个目标点
    (8.0, 8.0),    # 第二个目标点
    (0.0, 8.0),    # 第三个目标点
    # ... 添加更多航点
]
```

**地图配置**：
```python
self.slam = SLAMController(
    map_width=60.0,           # 增大地图范围
    map_height=60.0,
    map_resolution=0.2,       # 调整分辨率（越小越精细）
    map_origin=(-30.0, -30.0)
)
```

**控制参数**：
```python
self.slam.max_velocity = 2.0        # 最大速度 (m/s)
self.slam.max_yaw_rate = 1.0        # 最大偏航角速度 (rad/s)
self.slam.position_gain = 1.0       # 位置控制增益
```

### 算法原理

#### SLAM建图流程

1. **数据采集**：从激光雷达获取点云数据
2. **坐标转换**：将传感器坐标系转换到世界坐标系
3. **射线追踪**：使用Bresenham算法追踪从机器人到障碍物的射线
4. **地图更新**：
   - 射线路径上的单元格标记为空闲
   - 射线终点标记为占据
   - 使用对数几率进行概率更新

#### A*路径规划流程

1. **初始化**：将起点加入开放列表
2. **循环搜索**：
   - 从开放列表选择f值最小的节点
   - 检查是否到达目标
   - 扩展相邻节点
   - 更新g值和f值
3. **路径重建**：从目标回溯到起点
4. **路径优化**：平滑路径减少转折

#### 路径跟踪控制

使用比例控制器：
```
v_desired = K_p * (waypoint - current_position)
yaw_rate = K_y * (desired_yaw - current_yaw)
```

### 性能优化建议

1. **地图分辨率**：
   - 较低分辨率（0.2-0.5m）：快速规划，较少内存
   - 较高分辨率（0.05-0.1m）：更精确，更多计算

2. **激光雷达频率**：
   - 10-20 Hz：适用于大多数场景
   - 更高频率：更详细的地图，但计算量增加

3. **路径规划频率**：
   - 仅在新目标或环境变化时重新规划
   - 使用路径缓存

### 扩展功能

可以在此基础上添加：

1. **3D SLAM**：扩展到三维占据栅格
2. **多传感器融合**：结合相机、IMU等
3. **回环检测**：提高地图一致性
4. **动态障碍物处理**：实时避障
5. **地图合并**：多机器人协同建图

### 故障排除

**问题1**：无法规划路径
- 检查起点和终点是否在地图范围内
- 检查是否有可行路径（没有被障碍物完全阻隔）
- 增加障碍物膨胀半径

**问题2**：地图更新缓慢
- 减少地图分辨率
- 降低激光雷达频率
- 优化射线追踪算法

**问题3**：路径不平滑
- 增加路径平滑窗口大小
- 使用更高级的平滑算法（如B样条）

### 参考资料

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths

---

## English Documentation

### Project Overview

This is a complete SLAM (Simultaneous Localization and Mapping) and path planning project developed on Pegasus Simulator. The project demonstrates real-time map building using lidar sensors and autonomous navigation using the A* algorithm.

### Main Features

1. **Real-time SLAM Mapping**
   - 2D occupancy grid mapping using lidar data
   - Probabilistic map updates using log-odds
   - Map saving and loading support

2. **A* Path Planning**
   - A* search algorithm on occupancy grids
   - Diagonal movement support (8-connected)
   - Obstacle inflation for safety margins
   - Path smoothing algorithm

3. **Autonomous Navigation**
   - Path following controller
   - Real-time obstacle avoidance
   - Multi-waypoint navigation

### Project Structure

```
slam_path_planning/
├── occupancy_grid.py      # Occupancy grid implementation
├── path_planner.py        # A* path planning algorithm
├── slam_controller.py     # SLAM controller (integrates mapping and navigation)
└── README.md             # This document

slam_path_planning_example.py  # Main example application
```

### Core Components

#### 1. OccupancyGrid

**File**: `occupancy_grid.py`

The occupancy grid is the core data structure for the SLAM system, representing the environment map.

**Key Features**:
- Log-odds representation of occupancy probability
- Lidar data update support
- Bresenham ray tracing algorithm
- Probability and binary map conversion

**Key Parameters**:
```python
OccupancyGrid(
    width=50.0,        # Map width (meters)
    height=50.0,       # Map height (meters)
    resolution=0.1,    # Grid resolution (meters/cell)
    origin=(0.0, 0.0)  # Map origin coordinates
)
```

#### 2. AStarPlanner

**File**: `path_planner.py`

Implementation of the classic A* search algorithm for finding optimal paths in occupancy grids.

**Key Features**:
- 4-connected or 8-connected movement
- Euclidean heuristic function
- Obstacle inflation
- Path smoothing

#### 3. SLAMController

**File**: `slam_controller.py`

High-level controller integrating SLAM mapping, path planning, and motion control.

**Key Features**:
- Real-time map updates
- Automatic path planning
- Path tracking control
- Exploration goal generation
- Map saving

### Usage

#### Prerequisites

1. Install Isaac Sim 5.1.0 or higher
2. Install Pegasus Simulator
3. Python 3.7+
4. Dependencies: numpy, scipy

#### Running the Example

```bash
# Make sure Isaac Sim environment variables are set
# Run the SLAM path planning example
isaac_run examples/slam_path_planning_example.py
```

#### Custom Configuration

Modify the following parameters in `slam_path_planning_example.py`:

**Waypoint Configuration**:
```python
waypoints = [
    (8.0, 0.0),    # First waypoint
    (8.0, 8.0),    # Second waypoint
    (0.0, 8.0),    # Third waypoint
    # ... add more waypoints
]
```

**Map Configuration**:
```python
self.slam = SLAMController(
    map_width=60.0,           # Increase map size
    map_height=60.0,
    map_resolution=0.2,       # Adjust resolution (smaller = finer)
    map_origin=(-30.0, -30.0)
)
```

### Algorithm Details

#### SLAM Mapping Process

1. **Data Acquisition**: Get point cloud from lidar
2. **Coordinate Transform**: Convert sensor frame to world frame
3. **Ray Tracing**: Use Bresenham algorithm to trace rays from robot to obstacles
4. **Map Update**:
   - Mark cells along ray as free
   - Mark ray endpoint as occupied
   - Update probabilities using log-odds

#### A* Path Planning Process

1. **Initialize**: Add start node to open list
2. **Search Loop**:
   - Select node with minimum f-value from open list
   - Check if goal reached
   - Expand neighboring nodes
   - Update g-values and f-values
3. **Path Reconstruction**: Backtrack from goal to start
4. **Path Optimization**: Smooth path to reduce turns

### Performance Optimization

1. **Map Resolution**:
   - Low resolution (0.2-0.5m): Fast planning, less memory
   - High resolution (0.05-0.1m): More accurate, more computation

2. **Lidar Frequency**:
   - 10-20 Hz: Suitable for most scenarios
   - Higher frequency: More detailed map, but increased computation

3. **Path Planning Frequency**:
   - Replan only on new goals or environment changes
   - Use path caching

### Troubleshooting

**Issue 1**: Cannot plan path
- Check if start and goal are within map bounds
- Check if there's a feasible path (not completely blocked)
- Increase obstacle inflation radius

**Issue 2**: Slow map updates
- Reduce map resolution
- Lower lidar frequency
- Optimize ray tracing algorithm

**Issue 3**: Path not smooth
- Increase path smoothing window size
- Use advanced smoothing (e.g., B-splines)

### References

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths

## License

BSD-3-Clause License. Copyright (c) 2024. All rights reserved.
