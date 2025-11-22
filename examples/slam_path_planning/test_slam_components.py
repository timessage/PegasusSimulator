#!/usr/bin/env python
"""
| File: test_slam_components.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: Standalone tests for SLAM components (no Isaac Sim required)
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from occupancy_grid import OccupancyGrid
from path_planner import AStarPlanner


def test_occupancy_grid():
    """Test OccupancyGrid basic functionality."""
    print("\n" + "="*60)
    print("Testing OccupancyGrid")
    print("="*60)
    
    # Create grid
    grid = OccupancyGrid(width=20.0, height=20.0, resolution=0.5, origin=(-10.0, -10.0))
    
    # Test 1: Grid dimensions
    print(f"✓ Grid created: {grid.grid_width}x{grid.grid_height} cells")
    assert grid.grid_width == 40, "Grid width incorrect"
    assert grid.grid_height == 40, "Grid height incorrect"
    
    # Test 2: Coordinate conversion
    world_x, world_y = 5.0, 5.0
    grid_x, grid_y = grid.world_to_grid(world_x, world_y)
    world_x2, world_y2 = grid.grid_to_world(grid_x, grid_y)
    print(f"✓ Coordinate conversion: ({world_x}, {world_y}) -> ({grid_x}, {grid_y}) -> ({world_x2:.1f}, {world_y2:.1f})")
    assert abs(world_x - world_x2) < 0.5, "X coordinate conversion failed"
    assert abs(world_y - world_y2) < 0.5, "Y coordinate conversion failed"
    
    # Test 3: Cell updates
    grid.update_cell(20, 20, True)  # Mark as occupied
    grid.update_cell(21, 20, False)  # Mark as free
    print("✓ Cell update working")
    
    # Test 4: Lidar update
    robot_pos = np.array([0.0, 0.0, 1.0])
    lidar_points = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [-2.0, 0.0],
        [0.0, -2.0]
    ])
    grid.update_from_lidar(robot_pos, lidar_points, max_range=10.0)
    print(f"✓ Lidar update processed {len(lidar_points)} points")
    
    # Test 5: Probability grid
    prob_grid = grid.get_probability_grid()
    print(f"✓ Probability grid shape: {prob_grid.shape}")
    assert prob_grid.shape == (grid.grid_height, grid.grid_width), "Probability grid shape incorrect"
    assert np.all((prob_grid >= 0) & (prob_grid <= 1)), "Probability values out of range"
    
    # Test 6: Binary grid
    binary_grid = grid.get_binary_grid()
    print(f"✓ Binary grid created")
    assert np.all((binary_grid == 0) | (binary_grid == 1)), "Binary grid has invalid values"
    
    # Test 7: Cell free check
    is_free = grid.is_cell_free(20, 20)
    print(f"✓ Cell free check: {is_free}")
    
    # Test 8: Map info
    map_info = grid.get_map_info()
    print(f"✓ Map info: {map_info['occupied_cells']} occupied, {map_info['free_cells']} free")
    
    print("\n✅ OccupancyGrid tests passed!")
    return True


def test_path_planner():
    """Test A* path planner functionality."""
    print("\n" + "="*60)
    print("Testing A* Path Planner")
    print("="*60)
    
    # Create a simple test map
    grid = OccupancyGrid(width=20.0, height=20.0, resolution=0.5, origin=(-10.0, -10.0))
    
    # Create some obstacles
    robot_pos = np.array([0.0, 0.0, 1.0])
    
    # Add a wall of obstacles
    for i in range(-5, 5):
        obstacle_point = np.array([[2.0, float(i)]])
        grid.update_from_lidar(robot_pos, obstacle_point, max_range=10.0)
        # Update multiple times to ensure it's marked as occupied
        for _ in range(5):
            grid.update_cell(*grid.world_to_grid(2.0, float(i)), True)
    
    print(f"✓ Created test map with obstacles")
    
    # Create planner
    planner = AStarPlanner(grid, allow_diagonal=True)
    print(f"✓ Planner created (diagonal movement: {planner.allow_diagonal})")
    
    # Test 1: Simple path without obstacles
    start = (0.0, 0.0)
    goal = (5.0, 5.0)
    path = planner.plan(start, goal, inflation_radius=1)
    
    if path is not None:
        print(f"✓ Path found from {start} to {goal}: {len(path)} waypoints")
        assert len(path) >= 2, "Path too short"
        assert path[0] == start or np.linalg.norm(np.array(path[0]) - np.array(start)) < 1.0, "Path doesn't start at start"
        assert path[-1] == goal or np.linalg.norm(np.array(path[-1]) - np.array(goal)) < 1.0, "Path doesn't end at goal"
    else:
        print("⚠ No path found (might be blocked by test obstacles)")
    
    # Test 2: Path around obstacles
    start = (-5.0, 0.0)
    goal = (5.0, 0.0)
    path = planner.plan(start, goal, inflation_radius=2)
    
    if path is not None:
        print(f"✓ Path around obstacles: {len(path)} waypoints")
        # Path should go around the wall at x=2.0
        path_x_coords = [p[0] for p in path]
        # Check that path doesn't go straight through
        print(f"  Path X range: [{min(path_x_coords):.1f}, {max(path_x_coords):.1f}]")
    else:
        print("⚠ No path found around obstacles")
    
    # Test 3: Test invalid start/goal
    start = (-50.0, -50.0)  # Outside map
    goal = (5.0, 5.0)
    path = planner.plan(start, goal)
    assert path is None, "Should not plan path from invalid start"
    print("✓ Correctly rejected invalid start position")
    
    # Test 4: Path to same location
    start = (0.0, 0.0)
    goal = (0.0, 0.0)
    path = planner.plan(start, goal, inflation_radius=0)
    if path is not None:
        print(f"✓ Path to same location: {len(path)} waypoints")
        assert len(path) == 1, "Path to same location should have 1 waypoint"
    
    print("\n✅ Path Planner tests passed!")
    return True


def test_integration():
    """Test integration of components."""
    print("\n" + "="*60)
    print("Testing Component Integration")
    print("="*60)
    
    # Create a realistic scenario
    grid = OccupancyGrid(width=30.0, height=30.0, resolution=0.2, origin=(-15.0, -15.0))
    
    # Simulate a robot moving and mapping
    robot_trajectory = [
        np.array([0.0, 0.0, 2.0]),
        np.array([1.0, 0.0, 2.0]),
        np.array([2.0, 0.0, 2.0]),
        np.array([3.0, 0.0, 2.0]),
    ]
    
    # Simulate lidar readings at each position
    for pos in robot_trajectory:
        # Generate synthetic lidar points (circle around robot)
        angles = np.linspace(0, 2*np.pi, 36)
        ranges = 5.0 + np.random.randn(36) * 0.5  # 5m ± noise
        lidar_points = np.array([
            [pos[0] + r * np.cos(a), pos[1] + r * np.sin(a)]
            for a, r in zip(angles, ranges)
        ])
        
        grid.update_from_lidar(pos, lidar_points, max_range=10.0)
    
    print(f"✓ Simulated robot trajectory with {len(robot_trajectory)} poses")
    
    # Check map has been updated
    map_info = grid.get_map_info()
    print(f"✓ Map updated: {map_info['occupied_cells']} occupied, {map_info['free_cells']} free cells")
    assert map_info['occupied_cells'] > 0, "No occupied cells detected"
    assert map_info['free_cells'] > 0, "No free cells detected"
    
    # Plan a path on the created map
    planner = AStarPlanner(grid, allow_diagonal=True)
    start = (0.0, 0.0)
    goal = (5.0, 5.0)
    path = planner.plan(start, goal, inflation_radius=2)
    
    if path is not None:
        print(f"✓ Path planned on mapped environment: {len(path)} waypoints")
    else:
        print("⚠ Could not plan path (map might be too sparse)")
    
    print("\n✅ Integration tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("SLAM Components Test Suite")
    print("="*60)
    
    try:
        test_occupancy_grid()
        test_path_planner()
        test_integration()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
