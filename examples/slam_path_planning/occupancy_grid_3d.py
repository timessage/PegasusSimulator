#!/usr/bin/env python
"""
| File: occupancy_grid_3d.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: 3D occupancy grid (voxel grid) for 3D SLAM mapping
"""

import numpy as np
from typing import Tuple, List

class OccupancyGrid3D:
    """
    A 3D occupancy grid (voxel grid) for 3D SLAM applications.
    
    The grid uses a probabilistic approach where each voxel contains the log-odds
    of being occupied. This allows for efficient Bayesian updates.
    """
    
    def __init__(self, 
                 width: float = 50.0,
                 height: float = 50.0,
                 depth: float = 10.0,
                 resolution: float = 0.2,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize the 3D occupancy grid.
        
        Args:
            width: Width of the map in meters (X)
            height: Height of the map in meters (Y)
            depth: Depth of the map in meters (Z)
            resolution: Voxel size in meters
            origin: Origin of the map (x, y, z) in world coordinates
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.resolution = resolution
        self.origin = np.array(origin)
        
        # Calculate grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid_depth = int(depth / resolution)
        
        # Initialize grid with unknown state (log-odds = 0)
        # Positive values = occupied, negative = free, 0 = unknown
        self.grid = np.zeros((self.grid_depth, self.grid_height, self.grid_width), dtype=np.float32)
        
        # Parameters for log-odds updates
        self.log_odds_occ = 0.7  # Log-odds increase for occupied
        self.log_odds_free = -0.4  # Log-odds decrease for free
        self.log_odds_max = 10.0  # Maximum log-odds value
        self.log_odds_min = -10.0  # Minimum log-odds value
        
    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Convert world coordinates to grid indices.
        
        Args:
            x, y, z: World coordinates
            
        Returns:
            (grid_x, grid_y, grid_z): Grid indices
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        grid_z = int((z - self.origin[2]) / self.resolution)
        return grid_x, grid_y, grid_z
    
    def grid_to_world(self, grid_x: int, grid_y: int, grid_z: int) -> Tuple[float, float, float]:
        """
        Convert grid indices to world coordinates (voxel center).
        
        Args:
            grid_x, grid_y, grid_z: Grid indices
            
        Returns:
            (x, y, z): World coordinates
        """
        x = self.origin[0] + (grid_x + 0.5) * self.resolution
        y = self.origin[1] + (grid_y + 0.5) * self.resolution
        z = self.origin[2] + (grid_z + 0.5) * self.resolution
        return x, y, z
    
    def is_valid_cell(self, grid_x: int, grid_y: int, grid_z: int) -> bool:
        """
        Check if grid indices are within bounds.
        
        Args:
            grid_x, grid_y, grid_z: Grid indices
            
        Returns:
            True if cell is within grid bounds
        """
        return (0 <= grid_x < self.grid_width and 
                0 <= grid_y < self.grid_height and
                0 <= grid_z < self.grid_depth)
    
    def update_cell(self, grid_x: int, grid_y: int, grid_z: int, is_occupied: bool):
        """
        Update a single voxel in the grid using log-odds.
        
        Args:
            grid_x, grid_y, grid_z: Grid indices
            is_occupied: True if voxel is occupied, False if free
        """
        if not self.is_valid_cell(grid_x, grid_y, grid_z):
            return
        
        # Update log-odds
        if is_occupied:
            self.grid[grid_z, grid_y, grid_x] += self.log_odds_occ
        else:
            self.grid[grid_z, grid_y, grid_x] += self.log_odds_free
        
        # Clamp to min/max values
        self.grid[grid_z, grid_y, grid_x] = np.clip(
            self.grid[grid_z, grid_y, grid_x], 
            self.log_odds_min, 
            self.log_odds_max
        )
    
    def update_from_lidar(self, 
                         robot_position: np.ndarray,
                         lidar_points: np.ndarray,
                         max_range: float = 10.0):
        """
        Update the grid from a 3D lidar scan using ray tracing.
        
        Args:
            robot_position: Robot position [x, y, z] in world coordinates
            lidar_points: Nx3 array of lidar points in world coordinates
            max_range: Maximum valid range for lidar readings
        """
        robot_grid = self.world_to_grid(robot_position[0], robot_position[1], robot_position[2])
        
        if not self.is_valid_cell(robot_grid[0], robot_grid[1], robot_grid[2]):
            return
        
        # Process each lidar point
        for point in lidar_points:
            # Calculate distance
            distance = np.linalg.norm(point - robot_position)
            
            # Skip invalid readings
            if distance > max_range or distance < 0.1:
                continue
            
            # Get endpoint grid coordinates
            end_grid = self.world_to_grid(point[0], point[1], point[2])
            
            if not self.is_valid_cell(end_grid[0], end_grid[1], end_grid[2]):
                continue
            
            # Trace ray from robot to endpoint
            cells = self._bresenham_line_3d(
                robot_grid[0], robot_grid[1], robot_grid[2],
                end_grid[0], end_grid[1], end_grid[2]
            )
            
            # Mark all cells along the ray as free (except the last one)
            for i, (cell_x, cell_y, cell_z) in enumerate(cells[:-1]):
                if self.is_valid_cell(cell_x, cell_y, cell_z):
                    self.update_cell(cell_x, cell_y, cell_z, False)
            
            # Mark the endpoint as occupied
            if len(cells) > 0:
                end_x, end_y, end_z = cells[-1]
                if self.is_valid_cell(end_x, end_y, end_z):
                    self.update_cell(end_x, end_y, end_z, True)
    
    def _bresenham_line_3d(self, x0: int, y0: int, z0: int, 
                          x1: int, y1: int, z1: int) -> List[Tuple[int, int, int]]:
        """
        3D Bresenham's line algorithm for ray tracing.
        
        Args:
            x0, y0, z0: Start grid coordinates
            x1, y1, z1: End grid coordinates
            
        Returns:
            List of (x, y, z) grid coordinates along the line
        """
        cells = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        sz = 1 if z0 < z1 else -1
        
        # Determine the dominant direction
        if dx >= dy and dx >= dz:
            # X is dominant
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            
            while x0 != x1:
                cells.append((x0, y0, z0))
                x0 += sx
                if p1 >= 0:
                    y0 += sy
                    p1 -= 2 * dx
                if p2 >= 0:
                    z0 += sz
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                
        elif dy >= dx and dy >= dz:
            # Y is dominant
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            
            while y0 != y1:
                cells.append((x0, y0, z0))
                y0 += sy
                if p1 >= 0:
                    x0 += sx
                    p1 -= 2 * dy
                if p2 >= 0:
                    z0 += sz
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                
        else:
            # Z is dominant
            p1 = 2 * dx - dz
            p2 = 2 * dy - dz
            
            while z0 != z1:
                cells.append((x0, y0, z0))
                z0 += sz
                if p1 >= 0:
                    x0 += sx
                    p1 -= 2 * dz
                if p2 >= 0:
                    y0 += sy
                    p2 -= 2 * dz
                p1 += 2 * dx
                p2 += 2 * dy
        
        cells.append((x1, y1, z1))
        return cells
    
    def get_probability_grid(self) -> np.ndarray:
        """
        Convert log-odds grid to probability grid.
        
        Returns:
            Grid with probabilities [0, 1] where 0.5 is unknown
        """
        # Convert log-odds to probability: p = 1 - 1/(1 + exp(log_odds))
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(self.grid))
        return prob_grid
    
    def get_binary_grid(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary occupancy grid (0 = free, 1 = occupied).
        
        Args:
            threshold: Probability threshold for occupied cells
            
        Returns:
            Binary grid
        """
        prob_grid = self.get_probability_grid()
        binary_grid = (prob_grid > threshold).astype(np.uint8)
        return binary_grid
    
    def is_cell_free(self, grid_x: int, grid_y: int, grid_z: int, 
                    threshold: float = 0.3) -> bool:
        """
        Check if a voxel is free (safe for navigation).
        
        Args:
            grid_x, grid_y, grid_z: Grid indices
            threshold: Probability threshold (cell is free if p < threshold)
            
        Returns:
            True if voxel is free
        """
        if not self.is_valid_cell(grid_x, grid_y, grid_z):
            return False
        
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.grid[grid_z, grid_y, grid_x]))
        return prob < threshold
    
    def project_to_2d(self, method: str = 'max') -> np.ndarray:
        """
        Project 3D grid to 2D occupancy grid.
        
        Args:
            method: Projection method ('max', 'mean', 'min')
                   'max': Maximum occupancy along Z axis
                   'mean': Mean occupancy along Z axis
                   'min': Minimum occupancy along Z axis
            
        Returns:
            2D occupancy grid (height x width)
        """
        prob_grid_3d = self.get_probability_grid()
        
        if method == 'max':
            grid_2d = np.max(prob_grid_3d, axis=0)
        elif method == 'mean':
            grid_2d = np.mean(prob_grid_3d, axis=0)
        elif method == 'min':
            grid_2d = np.min(prob_grid_3d, axis=0)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return grid_2d
    
    def get_occupied_voxels(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get coordinates of all occupied voxels.
        
        Args:
            threshold: Probability threshold for occupied
            
        Returns:
            Nx3 array of world coordinates of occupied voxels
        """
        prob_grid = self.get_probability_grid()
        occupied_indices = np.argwhere(prob_grid > threshold)
        
        # Convert to world coordinates
        occupied_world = []
        for idx in occupied_indices:
            z, y, x = idx
            world_coord = self.grid_to_world(x, y, z)
            occupied_world.append(world_coord)
        
        return np.array(occupied_world) if occupied_world else np.zeros((0, 3))
    
    def get_map_info(self) -> dict:
        """
        Get map information for visualization and debugging.
        
        Returns:
            Dictionary with map information
        """
        prob_grid = self.get_probability_grid()
        return {
            'width': self.width,
            'height': self.height,
            'depth': self.depth,
            'resolution': self.resolution,
            'origin': self.origin,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'grid_depth': self.grid_depth,
            'occupied_voxels': np.sum(prob_grid > 0.5),
            'free_voxels': np.sum(prob_grid < 0.3),
            'unknown_voxels': np.sum((prob_grid >= 0.3) & (prob_grid <= 0.5)),
            'total_voxels': self.grid_width * self.grid_height * self.grid_depth
        }
    
    def save(self, filename: str):
        """
        Save the 3D occupancy grid to file.
        
        Args:
            filename: Output filename (.npz format)
        """
        np.savez(filename,
                grid=self.grid,
                width=self.width,
                height=self.height,
                depth=self.depth,
                resolution=self.resolution,
                origin=self.origin)
        print(f"3D occupancy grid saved to {filename}")
