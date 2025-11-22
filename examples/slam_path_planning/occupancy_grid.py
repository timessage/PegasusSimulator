#!/usr/bin/env python
"""
| File: occupancy_grid.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: Implementation of a 2D occupancy grid for SLAM mapping
"""

import numpy as np
from typing import Tuple, List

class OccupancyGrid:
    """
    A 2D occupancy grid map for SLAM applications.
    
    The grid uses a probabilistic approach where each cell contains the log-odds
    of being occupied. This allows for efficient Bayesian updates.
    """
    
    def __init__(self, 
                 width: float = 50.0,
                 height: float = 50.0, 
                 resolution: float = 0.1,
                 origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize the occupancy grid.
        
        Args:
            width: Width of the map in meters
            height: Height of the map in meters
            resolution: Cell size in meters
            origin: Origin of the map (x, y) in world coordinates
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin)
        
        # Calculate grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Initialize grid with unknown state (log-odds = 0)
        # Positive values = occupied, negative = free, 0 = unknown
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Parameters for log-odds updates
        self.log_odds_occ = 0.7  # Log-odds increase for occupied
        self.log_odds_free = -0.4  # Log-odds decrease for free
        self.log_odds_max = 10.0  # Maximum log-odds value
        self.log_odds_min = -10.0  # Minimum log-odds value
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid indices.
        
        Args:
            x, y: World coordinates
            
        Returns:
            (grid_x, grid_y): Grid indices
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates (cell center).
        
        Args:
            grid_x, grid_y: Grid indices
            
        Returns:
            (x, y): World coordinates
        """
        x = self.origin[0] + (grid_x + 0.5) * self.resolution
        y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return x, y
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if grid indices are within bounds.
        
        Args:
            grid_x, grid_y: Grid indices
            
        Returns:
            True if cell is within grid bounds
        """
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height
    
    def update_cell(self, grid_x: int, grid_y: int, is_occupied: bool):
        """
        Update a single cell in the grid using log-odds.
        
        Args:
            grid_x, grid_y: Grid indices
            is_occupied: True if cell is occupied, False if free
        """
        if not self.is_valid_cell(grid_x, grid_y):
            return
        
        # Update log-odds
        if is_occupied:
            self.grid[grid_y, grid_x] += self.log_odds_occ
        else:
            self.grid[grid_y, grid_x] += self.log_odds_free
        
        # Clamp to min/max values
        self.grid[grid_y, grid_x] = np.clip(
            self.grid[grid_y, grid_x], 
            self.log_odds_min, 
            self.log_odds_max
        )
    
    def update_from_lidar(self, 
                         robot_position: np.ndarray,
                         lidar_points: np.ndarray,
                         max_range: float = 10.0):
        """
        Update the grid from a lidar scan using ray tracing.
        
        Args:
            robot_position: Robot position [x, y] in world coordinates
            lidar_points: Nx2 array of lidar points in world coordinates
            max_range: Maximum valid range for lidar readings
        """
        robot_grid = self.world_to_grid(robot_position[0], robot_position[1])
        
        if not self.is_valid_cell(robot_grid[0], robot_grid[1]):
            return
        
        # Process each lidar point
        for point in lidar_points:
            # Calculate distance
            distance = np.linalg.norm(point - robot_position[:2])
            
            # Skip invalid readings
            if distance > max_range or distance < 0.1:
                continue
            
            # Get endpoint grid coordinates
            end_grid = self.world_to_grid(point[0], point[1])
            
            if not self.is_valid_cell(end_grid[0], end_grid[1]):
                continue
            
            # Trace ray from robot to endpoint
            cells = self._bresenham_line(robot_grid[0], robot_grid[1], 
                                         end_grid[0], end_grid[1])
            
            # Mark all cells along the ray as free (except the last one)
            for i, (cell_x, cell_y) in enumerate(cells[:-1]):
                if self.is_valid_cell(cell_x, cell_y):
                    self.update_cell(cell_x, cell_y, False)
            
            # Mark the endpoint as occupied
            if len(cells) > 0:
                end_x, end_y = cells[-1]
                if self.is_valid_cell(end_x, end_y):
                    self.update_cell(end_x, end_y, True)
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray tracing.
        
        Args:
            x0, y0: Start grid coordinates
            x1, y1: End grid coordinates
            
        Returns:
            List of (x, y) grid coordinates along the line
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
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
    
    def is_cell_free(self, grid_x: int, grid_y: int, threshold: float = 0.3) -> bool:
        """
        Check if a cell is free (safe for navigation).
        
        Args:
            grid_x, grid_y: Grid indices
            threshold: Probability threshold (cell is free if p < threshold)
            
        Returns:
            True if cell is free
        """
        if not self.is_valid_cell(grid_x, grid_y):
            return False
        
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.grid[grid_y, grid_x]))
        return prob < threshold
    
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
            'resolution': self.resolution,
            'origin': self.origin,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'occupied_cells': np.sum(prob_grid > 0.5),
            'free_cells': np.sum(prob_grid < 0.3),
            'unknown_cells': np.sum((prob_grid >= 0.3) & (prob_grid <= 0.5))
        }
