#!/usr/bin/env python
"""
| File: path_planner_3d.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: 3D A* path planning algorithm for aerial vehicle navigation
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from occupancy_grid_3d import OccupancyGrid3D

class AStarPlanner3D:
    """
    3D A* path planner for aerial vehicle navigation in 3D space.
    
    Implements the A* search algorithm to find optimal paths in a 3D occupancy grid.
    """
    
    def __init__(self, 
                 occupancy_grid: OccupancyGrid3D, 
                 allow_diagonal: bool = True):
        """
        Initialize the 3D A* path planner.
        
        Args:
            occupancy_grid: OccupancyGrid3D instance
            allow_diagonal: Whether to allow diagonal movements
        """
        self.grid = occupancy_grid
        self.allow_diagonal = allow_diagonal
        
        # Define movement directions
        if allow_diagonal:
            # 26-connected: all adjacent voxels including diagonals
            self.movements = []
            self.movement_costs = []
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        self.movements.append((dx, dy, dz))
                        
                        # Calculate cost based on Euclidean distance
                        cost = np.sqrt(dx**2 + dy**2 + dz**2)
                        self.movement_costs.append(cost)
        else:
            # 6-connected: only face neighbors (up, down, left, right, forward, backward)
            self.movements = [
                (1, 0, 0), (-1, 0, 0),   # X axis
                (0, 1, 0), (0, -1, 0),   # Y axis
                (0, 0, 1), (0, 0, -1)    # Z axis
            ]
            self.movement_costs = [1.0] * 6
    
    def plan(self, 
             start_pos: Tuple[float, float, float], 
             goal_pos: Tuple[float, float, float],
             inflation_radius: int = 2,
             max_iterations: int = 100000) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan a 3D path from start to goal using A*.
        
        Args:
            start_pos: Start position (x, y, z) in world coordinates
            goal_pos: Goal position (x, y, z) in world coordinates
            inflation_radius: Number of voxels to inflate obstacles (for safety margin)
            max_iterations: Maximum number of search iterations
            
        Returns:
            List of waypoints (x, y, z) in world coordinates, or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start_pos[0], start_pos[1], start_pos[2])
        goal_grid = self.grid.world_to_grid(goal_pos[0], goal_pos[1], goal_pos[2])
        
        # Check if start and goal are valid
        if not self.grid.is_valid_cell(start_grid[0], start_grid[1], start_grid[2]):
            print(f"Start position {start_pos} is outside the grid")
            return None
        
        if not self.grid.is_valid_cell(goal_grid[0], goal_grid[1], goal_grid[2]):
            print(f"Goal position {goal_pos} is outside the grid")
            return None
        
        if not self.grid.is_cell_free(start_grid[0], start_grid[1], start_grid[2]):
            print(f"Start position {start_pos} is occupied")
            return None
        
        if not self.grid.is_cell_free(goal_grid[0], goal_grid[1], goal_grid[2]):
            print(f"Goal position {goal_pos} is occupied")
            return None
        
        # Create inflated occupancy grid for safety
        inflated_grid = self._inflate_obstacles(inflation_radius)
        
        # Run A* search
        path_grid = self._astar_search(start_grid, goal_grid, inflated_grid, max_iterations)
        
        if path_grid is None:
            print(f"No path found from {start_pos} to {goal_pos}")
            return None
        
        # Convert grid path to world coordinates
        path_world = []
        for grid_x, grid_y, grid_z in path_grid:
            world_x, world_y, world_z = self.grid.grid_to_world(grid_x, grid_y, grid_z)
            path_world.append((world_x, world_y, world_z))
        
        # Smooth the path
        path_world = self._smooth_path(path_world)
        
        return path_world
    
    def _astar_search(self, 
                     start: Tuple[int, int, int], 
                     goal: Tuple[int, int, int],
                     inflated_grid: np.ndarray,
                     max_iterations: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        3D A* search algorithm implementation.
        
        Args:
            start: Start grid cell (x, y, z)
            goal: Goal grid cell (x, y, z)
            inflated_grid: Binary occupancy grid with inflated obstacles
            max_iterations: Maximum iterations
            
        Returns:
            List of grid cells (x, y, z) forming the path, or None if no path found
        """
        # Priority queue: (f_score, counter, current_node)
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        
        # Track visited nodes
        came_from = {}
        
        # Cost from start to each node
        g_score = {start: 0}
        
        # Estimated total cost (f = g + h)
        f_score = {start: self._heuristic(start, goal)}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == goal:
                print(f"Path found in {iterations} iterations")
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for i, (dx, dy, dz) in enumerate(self.movements):
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                
                # Check if neighbor is valid
                if not self.grid.is_valid_cell(neighbor[0], neighbor[1], neighbor[2]):
                    continue
                
                # Check if neighbor is occupied (in inflated grid)
                if inflated_grid[neighbor[2], neighbor[1], neighbor[0]] > 0:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self.movement_costs[i]
                
                # Update if we found a better path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        print(f"No path found after {iterations} iterations")
        # No path found
        return None
    
    def _heuristic(self, node: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
        """
        Heuristic function for A* (3D Euclidean distance).
        
        Args:
            node: Current node (x, y, z)
            goal: Goal node (x, y, z)
            
        Returns:
            Estimated distance to goal
        """
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        dz = abs(node[2] - goal[2])
        
        # 3D Euclidean distance
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _reconstruct_path(self, 
                         came_from: dict, 
                         current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Reconstruct the path from came_from dictionary.
        
        Args:
            came_from: Dictionary mapping nodes to their predecessors
            current: Goal node
            
        Returns:
            List of nodes forming the path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _inflate_obstacles(self, radius: int) -> np.ndarray:
        """
        Inflate obstacles by a given radius for safety margin.
        
        Args:
            radius: Inflation radius in grid cells
            
        Returns:
            Binary inflated grid
        """
        # Get binary occupancy grid
        binary_grid = self.grid.get_binary_grid(threshold=0.5)
        
        if radius <= 0:
            return binary_grid
        
        # Create inflated grid
        inflated = np.zeros_like(binary_grid)
        
        # For each occupied voxel, inflate around it
        occupied_voxels = np.argwhere(binary_grid > 0)
        
        for z, y, x in occupied_voxels:
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        # Check if within radius (using L-infinity norm for simplicity)
                        if abs(dx) <= radius and abs(dy) <= radius and abs(dz) <= radius:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if self.grid.is_valid_cell(nx, ny, nz):
                                inflated[nz, ny, nx] = 1
        
        return inflated
    
    def _smooth_path(self, 
                    path: List[Tuple[float, float, float]], 
                    window_size: int = 3) -> List[Tuple[float, float, float]]:
        """
        Smooth the 3D path using a simple moving average filter.
        
        Args:
            path: List of waypoints (x, y, z)
            window_size: Size of smoothing window
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Keep start and goal fixed
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Calculate window bounds
            start_idx = max(1, i - window_size // 2)
            end_idx = min(len(path) - 1, i + window_size // 2 + 1)
            
            # Average positions in window
            avg_x = np.mean([p[0] for p in path[start_idx:end_idx]])
            avg_y = np.mean([p[1] for p in path[start_idx:end_idx]])
            avg_z = np.mean([p[2] for p in path[start_idx:end_idx]])
            
            smoothed.append((avg_x, avg_y, avg_z))
        
        smoothed.append(path[-1])
        
        return smoothed
    
    def plan_safe_trajectory(self,
                            start_pos: Tuple[float, float, float],
                            goal_pos: Tuple[float, float, float],
                            preferred_altitude: float = 3.0,
                            altitude_weight: float = 0.5) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan a trajectory that prefers certain altitudes (e.g., for energy efficiency).
        
        Args:
            start_pos: Start position (x, y, z)
            goal_pos: Goal position (x, y, z)
            preferred_altitude: Preferred altitude for flight
            altitude_weight: Weight for altitude preference in cost function
            
        Returns:
            List of waypoints or None
        """
        # First try standard path planning
        path = self.plan(start_pos, goal_pos)
        
        if path is None:
            return None
        
        # Adjust altitudes in the path to prefer the specified altitude
        adjusted_path = []
        
        for i, (x, y, z) in enumerate(path):
            # Keep start and goal altitudes
            if i == 0 or i == len(path) - 1:
                adjusted_path.append((x, y, z))
            else:
                # Blend current altitude with preferred altitude
                adjusted_z = (1 - altitude_weight) * z + altitude_weight * preferred_altitude
                
                # Check if adjusted position is free
                grid_pos = self.grid.world_to_grid(x, y, adjusted_z)
                if self.grid.is_valid_cell(*grid_pos) and self.grid.is_cell_free(*grid_pos):
                    adjusted_path.append((x, y, adjusted_z))
                else:
                    # Keep original altitude if adjusted one is not free
                    adjusted_path.append((x, y, z))
        
        return adjusted_path
