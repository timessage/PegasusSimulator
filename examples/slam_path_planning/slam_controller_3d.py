#!/usr/bin/env python
"""
| File: slam_controller_3d.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: 3D SLAM controller with 3D path planning and point cloud management
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation
from .occupancy_grid_3d import OccupancyGrid3D
from .path_planner_3d import AStarPlanner3D
from .point_cloud_3d import PointCloud3D

class SLAMController3D:
    """
    3D SLAM controller that performs simultaneous localization, mapping, and navigation in 3D space.
    
    This controller:
    1. Maintains a 3D point cloud map
    2. Updates a 3D occupancy grid (voxel grid) from lidar data
    3. Plans 3D paths to goal positions using A*
    4. Controls the vehicle to follow planned 3D paths
    """
    
    def __init__(self, 
                 map_width: float = 50.0,
                 map_height: float = 50.0,
                 map_depth: float = 10.0,
                 map_resolution: float = 0.3,
                 map_origin: Tuple[float, float, float] = (-25.0, -25.0, 0.0)):
        """
        Initialize the 3D SLAM controller.
        
        Args:
            map_width: Width of the map in meters (X)
            map_height: Height of the map in meters (Y)
            map_depth: Depth of the map in meters (Z)
            map_resolution: Voxel size in meters
            map_origin: Origin of the map (x, y, z) in world coordinates
        """
        # Create 3D occupancy grid
        self.occupancy_grid = OccupancyGrid3D(
            width=map_width,
            height=map_height,
            depth=map_depth,
            resolution=map_resolution,
            origin=map_origin
        )
        
        # Create 3D point cloud for visualization
        self.point_cloud = PointCloud3D(
            voxel_size=map_resolution,
            max_points=500000
        )
        
        # Create 3D path planner
        self.path_planner = AStarPlanner3D(self.occupancy_grid, allow_diagonal=True)
        
        # Navigation state
        self.current_path = None
        self.current_waypoint_idx = 0
        self.goal_position = None
        self.path_following_distance = 1.5  # Distance threshold to consider waypoint reached
        
        # Control parameters
        self.max_velocity = 2.5  # Maximum velocity in m/s
        self.max_vertical_velocity = 1.5  # Maximum vertical velocity
        self.max_yaw_rate = 1.0  # Maximum yaw rate in rad/s
        self.position_gain = 1.2  # Proportional gain for position control
        self.altitude_gain = 1.0  # Proportional gain for altitude control
        self.yaw_gain = 2.0  # Proportional gain for yaw control
        
        # SLAM state
        self.mapping_enabled = True
        self.preferred_altitude = 3.0  # Preferred cruising altitude
        
    def update_map(self, position: np.ndarray, lidar_data: dict):
        """
        Update the 3D occupancy grid and point cloud from lidar sensor data.
        
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
            
            # Update 3D occupancy grid
            self.occupancy_grid.update_from_lidar(
                robot_position=position,
                lidar_points=points_3d,
                max_range=15.0
            )
            
            # Add to point cloud for visualization
            self.point_cloud.add_scan(points_3d, robot_position=position)
    
    def set_goal(self, goal_position: Tuple[float, float, float], 
                current_position: Tuple[float, float, float]) -> bool:
        """
        Set a new 3D goal position and plan a path.
        
        Args:
            goal_position: Goal position (x, y, z) in world coordinates
            current_position: Current position (x, y, z) in world coordinates
            
        Returns:
            True if path planning succeeded
        """
        self.goal_position = goal_position
        
        # Plan 3D path using A*
        path = self.path_planner.plan_safe_trajectory(
            start_pos=current_position,
            goal_pos=goal_position,
            preferred_altitude=self.preferred_altitude,
            altitude_weight=0.3
        )
        
        if path is None:
            print(f"Failed to plan 3D path to goal {goal_position}")
            return False
        
        self.current_path = path
        self.current_waypoint_idx = 0
        print(f"Planned 3D path with {len(path)} waypoints to goal {goal_position}")
        
        # Print altitude profile
        altitudes = [p[2] for p in path]
        print(f"  Altitude range: [{min(altitudes):.1f}, {max(altitudes):.1f}] m")
        
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
            print("3D path following completed!")
            self.current_path = None
            return control
        
        waypoint = self.current_path[self.current_waypoint_idx]
        
        # Calculate 3D distance to waypoint
        distance = np.linalg.norm(np.array(waypoint) - position)
        
        # Check if waypoint reached
        if distance < self.path_following_distance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.current_path):
                print(f"Reached 3D waypoint {self.current_waypoint_idx}/{len(self.current_path)}")
                waypoint = self.current_path[self.current_waypoint_idx]
            else:
                return control
        
        # Calculate desired velocity towards waypoint (3D)
        direction_3d = np.array(waypoint) - position
        distance_3d = np.linalg.norm(direction_3d)
        
        if distance_3d > 0.1:
            # Normalize direction
            direction_3d = direction_3d / distance_3d
            
            # Separate horizontal and vertical components
            horizontal_dir = direction_3d[:2]
            horizontal_dist = np.linalg.norm(horizontal_dir)
            
            # Calculate desired horizontal velocity
            if horizontal_dist > 0.1:
                horizontal_dir = horizontal_dir / horizontal_dist
                desired_horizontal_speed = min(self.max_velocity, 
                                              horizontal_dist * self.position_gain)
                desired_vx = horizontal_dir[0] * desired_horizontal_speed
                desired_vy = horizontal_dir[1] * desired_horizontal_speed
            else:
                desired_vx = 0.0
                desired_vy = 0.0
            
            # Calculate desired vertical velocity
            altitude_error = waypoint[2] - position[2]
            desired_vz = np.clip(altitude_error * self.altitude_gain,
                                -self.max_vertical_velocity,
                                self.max_vertical_velocity)
            
            control['velocity'] = np.array([desired_vx, desired_vy, desired_vz])
            
            # Calculate desired yaw (point towards horizontal direction)
            if horizontal_dist > 0.1:
                desired_yaw = np.arctan2(horizontal_dir[1], horizontal_dir[0])
                
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
    
    def get_visualization_data(self) -> dict:
        """
        Get data for visualizing the 3D map and path.
        
        Returns:
            Dictionary with visualization data
        """
        data = {
            'point_cloud_points': self.point_cloud.get_points(),
            'point_cloud_colors': self.point_cloud.get_colors(),
            'occupancy_grid_3d': self.occupancy_grid.get_probability_grid(),
            'occupancy_grid_2d': self.occupancy_grid.project_to_2d(method='max'),
            'map_info': self.occupancy_grid.get_map_info(),
            'point_cloud_stats': self.point_cloud.get_statistics(),
            'path_3d': self.current_path,
            'goal': self.goal_position
        }
        return data
    
    def save_map(self, filename_base: str):
        """
        Save the current 3D map and point cloud to files.
        
        Args:
            filename_base: Base filename (without extension)
        """
        # Save 3D occupancy grid
        self.occupancy_grid.save(f"{filename_base}_grid.npz")
        
        # Save point cloud
        self.point_cloud.save(f"{filename_base}_pointcloud.npz")
        
        # Export point cloud to PLY for external visualization
        self.point_cloud.export_ply(f"{filename_base}_pointcloud.ply")
        
        print(f"3D map saved with base name: {filename_base}")
    
    def get_obstacle_free_altitude(self, x: float, y: float, 
                                   min_altitude: float = 1.0,
                                   max_altitude: float = 8.0) -> float:
        """
        Find a safe altitude at a given (x, y) position.
        
        Args:
            x, y: Horizontal position
            min_altitude: Minimum altitude to check
            max_altitude: Maximum altitude to check
            
        Returns:
            Safe altitude, or preferred_altitude if no obstacles found
        """
        # Check altitudes from min to max
        for z in np.linspace(min_altitude, max_altitude, 20):
            grid_pos = self.occupancy_grid.world_to_grid(x, y, z)
            if (self.occupancy_grid.is_valid_cell(*grid_pos) and 
                self.occupancy_grid.is_cell_free(*grid_pos)):
                return z
        
        # Return preferred altitude if no free space found
        return self.preferred_altitude
    
    def replan_if_needed(self, current_position: Tuple[float, float, float],
                        lookahead_distance: float = 3.0) -> bool:
        """
        Check if replanning is needed due to obstacles in the path.
        
        Args:
            current_position: Current robot position (x, y, z)
            lookahead_distance: Distance to look ahead on the path
            
        Returns:
            True if replanning was performed
        """
        if self.current_path is None or self.goal_position is None:
            return False
        
        # Check next few waypoints for obstacles
        for i in range(self.current_waypoint_idx, 
                      min(self.current_waypoint_idx + 5, len(self.current_path))):
            waypoint = self.current_path[i]
            grid_pos = self.occupancy_grid.world_to_grid(*waypoint)
            
            # If waypoint is now occupied, replan
            if (self.occupancy_grid.is_valid_cell(*grid_pos) and 
                not self.occupancy_grid.is_cell_free(*grid_pos)):
                print("Obstacle detected in path! Replanning...")
                return self.set_goal(self.goal_position, current_position)
        
        return False
