#!/usr/bin/env python
"""
| File: slam_controller.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: SLAM controller that integrates mapping and path planning with vehicle control
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation
from .occupancy_grid import OccupancyGrid
from .path_planner import AStarPlanner

class SLAMController:
    """
    SLAM controller that performs simultaneous localization, mapping, and navigation.
    
    This controller:
    1. Updates the occupancy grid map from lidar data
    2. Plans paths to goal positions
    3. Controls the vehicle to follow planned paths
    """
    
    def __init__(self, 
                 map_width: float = 50.0,
                 map_height: float = 50.0,
                 map_resolution: float = 0.2,
                 map_origin: Tuple[float, float] = (-25.0, -25.0)):
        """
        Initialize the SLAM controller.
        
        Args:
            map_width: Width of the map in meters
            map_height: Height of the map in meters
            map_resolution: Grid cell size in meters
            map_origin: Origin of the map (x, y) in world coordinates
        """
        # Create occupancy grid
        self.occupancy_grid = OccupancyGrid(
            width=map_width,
            height=map_height,
            resolution=map_resolution,
            origin=map_origin
        )
        
        # Create path planner
        self.path_planner = AStarPlanner(self.occupancy_grid, allow_diagonal=True)
        
        # Navigation state
        self.current_path = None
        self.current_waypoint_idx = 0
        self.goal_position = None
        self.path_following_distance = 1.0  # Distance threshold to consider waypoint reached
        
        # Control parameters
        self.max_velocity = 2.0  # Maximum velocity in m/s
        self.max_yaw_rate = 1.0  # Maximum yaw rate in rad/s
        self.position_gain = 1.0  # Proportional gain for position control
        self.yaw_gain = 2.0  # Proportional gain for yaw control
        
        # SLAM state
        self.mapping_enabled = True
        self.min_altitude = 2.0  # Minimum altitude for navigation
        self.exploration_points = []  # Points to explore for mapping
        
    def update_map(self, position: np.ndarray, lidar_data: dict):
        """
        Update the occupancy grid map from lidar sensor data.
        
        Args:
            position: Robot position [x, y, z] in world coordinates
            lidar_data: Dictionary containing lidar scan data with keys:
                       'points': Nx3 array of 3D points in world coordinates
        """
        if not self.mapping_enabled:
            return
        
        # Extract lidar points
        if 'points' in lidar_data and lidar_data['points'] is not None:
            points_3d = lidar_data['points']
            
            # Project 3D points to 2D (ignore z coordinate)
            points_2d = points_3d[:, :2]
            
            # Update occupancy grid
            self.occupancy_grid.update_from_lidar(
                robot_position=position,
                lidar_points=points_2d,
                max_range=10.0
            )
    
    def set_goal(self, goal_position: Tuple[float, float]) -> bool:
        """
        Set a new goal position and plan a path.
        
        Args:
            goal_position: Goal position (x, y) in world coordinates
            
        Returns:
            True if path planning succeeded
        """
        self.goal_position = goal_position
        
        # Get current robot position (use map center as default)
        current_pos = (0.0, 0.0)  # Will be updated with actual position
        
        # Plan path using A*
        path = self.path_planner.plan(
            start_pos=current_pos,
            goal_pos=goal_position,
            inflation_radius=3  # Safety margin of 3 cells
        )
        
        if path is None:
            print(f"Failed to plan path to goal {goal_position}")
            return False
        
        self.current_path = path
        self.current_waypoint_idx = 0
        print(f"Planned path with {len(path)} waypoints to goal {goal_position}")
        
        return True
    
    def update(self, state: dict) -> dict:
        """
        Main update function called every timestep.
        
        Args:
            state: Dictionary containing vehicle state:
                  'position': [x, y, z] position
                  'orientation': [w, x, y, z] quaternion
                  'velocity': [vx, vy, vz] velocity
                  'lidar': dictionary with lidar data
                  
        Returns:
            Control command dictionary with:
                'velocity': [vx, vy, vz] desired velocity in world frame
                'yaw_rate': desired yaw rate in rad/s
        """
        # Extract state information
        position = np.array(state.get('position', [0, 0, 0]))
        orientation = state.get('orientation', [1, 0, 0, 0])  # [w, x, y, z]
        lidar_data = state.get('lidar', {})
        
        # Update map from lidar data
        self.update_map(position, lidar_data)
        
        # Default control: hover
        control = {
            'velocity': np.array([0.0, 0.0, 0.0]),
            'yaw_rate': 0.0
        }
        
        # Check if we have an active path to follow
        if self.current_path is None or len(self.current_path) == 0:
            return control
        
        # Get current waypoint
        if self.current_waypoint_idx >= len(self.current_path):
            # Path completed
            print("Path following completed!")
            self.current_path = None
            return control
        
        # Update current position in path planner
        current_pos_2d = (position[0], position[1])
        
        waypoint = self.current_path[self.current_waypoint_idx]
        
        # Calculate distance to waypoint
        distance = np.linalg.norm([waypoint[0] - position[0], 
                                   waypoint[1] - position[1]])
        
        # Check if waypoint reached
        if distance < self.path_following_distance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.current_path):
                print(f"Reached waypoint {self.current_waypoint_idx}/{len(self.current_path)}")
                waypoint = self.current_path[self.current_waypoint_idx]
            else:
                return control
        
        # Calculate desired velocity towards waypoint
        direction = np.array([waypoint[0] - position[0], 
                             waypoint[1] - position[1]])
        distance_2d = np.linalg.norm(direction)
        
        if distance_2d > 0.1:
            # Normalize direction
            direction = direction / distance_2d
            
            # Calculate desired velocity (proportional control)
            desired_speed = min(self.max_velocity, distance_2d * self.position_gain)
            desired_velocity = direction * desired_speed
            
            # Maintain altitude
            altitude_error = self.min_altitude - position[2]
            desired_vz = np.clip(altitude_error * 0.5, -0.5, 0.5)
            
            control['velocity'] = np.array([desired_velocity[0], 
                                           desired_velocity[1], 
                                           desired_vz])
            
            # Calculate desired yaw (point towards waypoint)
            desired_yaw = np.arctan2(direction[1], direction[0])
            
            # Get current yaw from quaternion
            rot = Rotation.from_quat([orientation[1], orientation[2], 
                                     orientation[3], orientation[0]])
            current_yaw = rot.as_euler('xyz')[2]
            
            # Calculate yaw error
            yaw_error = desired_yaw - current_yaw
            
            # Normalize yaw error to [-pi, pi]
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            
            # Calculate desired yaw rate
            control['yaw_rate'] = np.clip(yaw_error * self.yaw_gain, 
                                         -self.max_yaw_rate, 
                                         self.max_yaw_rate)
        
        return control
    
    def get_map_visualization_data(self) -> dict:
        """
        Get data for visualizing the map.
        
        Returns:
            Dictionary with visualization data:
                'probability_grid': 2D array of occupancy probabilities
                'path': Current path waypoints
                'goal': Current goal position
        """
        data = {
            'probability_grid': self.occupancy_grid.get_probability_grid(),
            'map_info': self.occupancy_grid.get_map_info(),
            'path': self.current_path,
            'goal': self.goal_position
        }
        return data
    
    def get_exploration_goal(self, current_position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Generate an exploration goal for mapping unknown areas.
        
        Args:
            current_position: Current robot position (x, y)
            
        Returns:
            Exploration goal position (x, y) or None
        """
        # Get probability grid
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Find frontier cells (free cells adjacent to unknown cells)
        frontier_cells = []
        
        for y in range(1, self.occupancy_grid.grid_height - 1):
            for x in range(1, self.occupancy_grid.grid_width - 1):
                # Check if cell is free
                if prob_grid[y, x] < 0.3:
                    # Check if adjacent to unknown cell
                    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if 0.45 <= prob_grid[ny, nx] <= 0.55:  # Unknown
                            frontier_cells.append((x, y))
                            break
        
        if not frontier_cells:
            return None
        
        # Convert to world coordinates
        frontier_points = []
        for grid_x, grid_y in frontier_cells:
            world_x, world_y = self.occupancy_grid.grid_to_world(grid_x, grid_y)
            frontier_points.append((world_x, world_y))
        
        # Select closest frontier point
        min_dist = float('inf')
        best_goal = None
        
        for point in frontier_points:
            dist = np.linalg.norm([point[0] - current_position[0],
                                  point[1] - current_position[1]])
            if dist < min_dist:
                min_dist = dist
                best_goal = point
        
        return best_goal
    
    def save_map(self, filename: str):
        """
        Save the current map to a file.
        
        Args:
            filename: Output filename (numpy .npz format)
        """
        np.savez(filename,
                grid=self.occupancy_grid.grid,
                width=self.occupancy_grid.width,
                height=self.occupancy_grid.height,
                resolution=self.occupancy_grid.resolution,
                origin=self.occupancy_grid.origin)
        print(f"Map saved to {filename}")
