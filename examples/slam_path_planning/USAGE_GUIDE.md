# SLAM使用指南 / SLAM Usage Guide

## 快速开始 / Quick Start

### 1. 运行3D SLAM示例 (推荐)

```bash
# 确保Isaac Sim环境已配置
source ~/.bashrc  # 或 ~/.zshrc

# 运行3D SLAM示例
cd /path/to/PegasusSimulator
isaac_run examples/slam_path_planning_3d_example.py
```

### 2. 运行2D SLAM示例

```bash
isaac_run examples/slam_path_planning_example.py
```

---

## 自定义配置 / Custom Configuration

### 修改航点 / Change Waypoints

编辑示例文件中的航点列表:

```python
# 3D航点 (x, y, z)
waypoints = [
    (8.0, 0.0, 3.0),    # 第一个目标
    (8.0, 8.0, 4.5),    # 第二个目标
    (0.0, 8.0, 3.5),    # 第三个目标
    # 添加更多...
]
```

### 调整地图参数 / Adjust Map Parameters

```python
slam = SLAMController3D(
    map_width=60.0,         # 地图宽度(米)
    map_height=60.0,        # 地图长度(米)
    map_depth=15.0,         # 地图高度(米)
    map_resolution=0.3,     # 体素大小(米)
    map_origin=(-30.0, -30.0, 0.0)  # 地图原点
)
```

### 修改控制参数 / Modify Control Parameters

```python
slam.max_velocity = 2.5              # 最大速度 (m/s)
slam.max_vertical_velocity = 1.5     # 最大垂直速度 (m/s)
slam.max_yaw_rate = 1.0              # 最大偏航角速度 (rad/s)
slam.position_gain = 1.2             # 位置控制增益
slam.altitude_gain = 1.0             # 高度控制增益
slam.preferred_altitude = 3.0        # 优选巡航高度 (m)
```

---

## 独立使用组件 / Using Components Independently

### 仅使用占据网格 / Occupancy Grid Only

```python
from occupancy_grid_3d import OccupancyGrid3D
import numpy as np

# 创建3D占据网格
grid = OccupancyGrid3D(
    width=50.0,
    height=50.0,
    depth=10.0,
    resolution=0.2,
    origin=(-25.0, -25.0, 0.0)
)

# 添加激光雷达数据
robot_pos = np.array([0.0, 0.0, 2.0])
lidar_points = np.array([
    [1.0, 0.0, 2.0],
    [0.0, 1.0, 2.0],
    [-1.0, 0.0, 2.0],
])
grid.update_from_lidar(robot_pos, lidar_points)

# 获取概率地图
prob_grid = grid.get_probability_grid()

# 保存地图
grid.save("my_map.npz")
```

### 仅使用路径规划 / Path Planning Only

```python
from occupancy_grid_3d import OccupancyGrid3D
from path_planner_3d import AStarPlanner3D

# 假设已有占据网格
grid = OccupancyGrid3D(...)

# 创建规划器
planner = AStarPlanner3D(grid, allow_diagonal=True)

# 规划路径
start = (0.0, 0.0, 2.0)
goal = (10.0, 10.0, 3.0)
path = planner.plan(start, goal, inflation_radius=3)

if path:
    print(f"找到路径，共{len(path)}个航点")
    for i, waypoint in enumerate(path):
        print(f"  {i+1}. {waypoint}")
```

### 仅使用点云 / Point Cloud Only

```python
from point_cloud_3d import PointCloud3D
import numpy as np

# 创建点云
pc = PointCloud3D(voxel_size=0.1, max_points=100000)

# 添加扫描数据
points = np.random.rand(1000, 3) * 10  # 随机点云
pc.add_scan(points)

# 获取统计信息
stats = pc.get_statistics()
print(f"点云包含 {stats['num_points']} 个点")

# 导出为PLY
pc.export_ply("my_pointcloud.ply")

# 保存点云
pc.save("my_pointcloud.npz")
```

### 可视化地图 / Visualize Maps

```python
from visualizer_3d import PointCloudVisualizer3D
from point_cloud_3d import PointCloud3D

# 加载点云
pc = PointCloud3D.load("my_pointcloud.npz")

# 创建可视化器
viz = PointCloudVisualizer3D()

# 可视化
viz.visualize_point_cloud(
    points=pc.get_points(),
    colors=pc.get_colors(),
    title="我的3D地图",
    save_path="visualization.png"
)

# 多视角可视化
viz.visualize_multiple_views(
    points=pc.get_points(),
    colors=pc.get_colors(),
    save_path="multiview.png"
)
```

---

## 高级用法 / Advanced Usage

### 动态重规划 / Dynamic Replanning

```python
# 在导航过程中
if slam.replan_if_needed(current_position):
    print("检测到障碍物，已重新规划路径")
```

### 寻找安全高度 / Find Safe Altitude

```python
# 在给定(x, y)位置找到安全高度
safe_z = slam.get_obstacle_free_altitude(
    x=5.0, 
    y=5.0,
    min_altitude=1.0,
    max_altitude=8.0
)
print(f"安全高度: {safe_z}米")
```

### 探索未知区域 / Explore Unknown Areas

```python
# 生成探索目标
current_pos = (0.0, 0.0)
exploration_goal = slam.get_exploration_goal(current_pos)

if exploration_goal:
    print(f"探索目标: {exploration_goal}")
    slam.set_goal(exploration_goal, current_pos)
```

### 优选高度路径规划 / Preferred Altitude Planning

```python
# 规划时优选特定高度(例如节能)
path = planner.plan_safe_trajectory(
    start_pos=(0, 0, 2),
    goal_pos=(10, 10, 2),
    preferred_altitude=3.0,      # 优选高度
    altitude_weight=0.5          # 高度优化权重
)
```

---

## 输出文件 / Output Files

### 运行后生成的文件

```
slam_path_planning/
├── saved_map_3d_grid.npz              # 3D体素网格
├── saved_map_3d_pointcloud.npz        # 点云数据
├── saved_map_3d_pointcloud.ply        # PLY格式点云
├── final_3d_map.png                   # 最终3D地图
├── final_3d_map_multiview.png         # 多视角视图
├── final_combined_view.png            # 组合视图
└── 3d_map_t*.png                      # 中间状态可视化
```

### 文件格式说明

**NPZ文件** (NumPy压缩格式):
```python
# 加载
data = np.load("saved_map_3d_grid.npz")
grid = data['grid']
resolution = data['resolution']
```

**PLY文件** (多边形文件格式):
- 可用CloudCompare、MeshLab等工具打开
- 包含点位置和颜色信息
- 标准ASCII格式

---

## 性能调优 / Performance Tuning

### 提高建图速度

```python
# 降低分辨率
slam = SLAMController3D(map_resolution=0.5)  # 更大的体素

# 减少最大点数
pc = PointCloud3D(max_points=100000)  # 更少的点
```

### 提高规划速度

```python
# 减少障碍物膨胀
path = planner.plan(start, goal, inflation_radius=1)

# 使用6连通而非26连通
planner = AStarPlanner3D(grid, allow_diagonal=False)
```

### 提高控制精度

```python
# 增加控制增益
slam.position_gain = 2.0
slam.altitude_gain = 1.5

# 减小航点到达阈值
slam.path_following_distance = 0.5  # 米
```

---

## 故障排除 / Troubleshooting

### 问题: 找不到路径

**解决方案**:
1. 检查起点和终点是否在地图范围内
2. 检查起点/终点是否被障碍物占据
3. 增加地图分辨率
4. 减少障碍物膨胀半径

```python
# 检查位置是否有效
grid_pos = grid.world_to_grid(x, y, z)
is_valid = grid.is_valid_cell(*grid_pos)
is_free = grid.is_cell_free(*grid_pos)
print(f"有效: {is_valid}, 空闲: {is_free}")
```

### 问题: 地图更新缓慢

**解决方案**:
1. 降低激光雷达频率
2. 增加地图分辨率(更大的体素)
3. 减少激光雷达点数

### 问题: 可视化失败

**解决方案**:
```python
# 检查点云是否为空
if len(pc.get_points()) == 0:
    print("警告: 点云为空")

# 尝试不同的matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
```

### 问题: 内存不足

**解决方案**:
```python
# 限制点云大小
pc = PointCloud3D(max_points=50000)

# 减小地图范围
slam = SLAMController3D(
    map_width=30.0,
    map_height=30.0,
    map_depth=8.0
)

# 增加体素大小
slam = SLAMController3D(map_resolution=0.5)
```

---

## 最佳实践 / Best Practices

### 1. 建图阶段

✅ 先建图后导航(两阶段方法)
✅ 覆盖整个工作区域
✅ 使用适当的飞行高度(2-5米)
✅ 保持平稳飞行速度

### 2. 路径规划

✅ 使用适当的安全边界(2-3个体素)
✅ 定期检查路径有效性
✅ 必要时重新规划
✅ 考虑高度优化

### 3. 导航控制

✅ 使用平滑的速度曲线
✅ 监控执行误差
✅ 实现故障安全机制
✅ 记录遥测数据

### 4. 数据管理

✅ 定期保存地图
✅ 导出可视化
✅ 备份关键数据
✅ 记录参数配置

---

## 示例代码库 / Example Code Repository

完整示例请参考:
- `slam_path_planning_example.py` - 2D示例
- `slam_path_planning_3d_example.py` - 3D示例
- `test_slam_components.py` - 单元测试

---

## 获取帮助 / Get Help

遇到问题？
1. 查看 README.md
2. 查看 FEATURES.md
3. 检查代码注释
4. 运行测试用例
5. 提交Issue

---

最后更新 / Last Updated: 2024
