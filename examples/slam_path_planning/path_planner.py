#!/usr/bin/env python
"""
| File: path_planner.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: Implementation of A* path planning algorithm for navigation
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from occupancy_grid import OccupancyGrid

class AStarPlanner:
    """
    A* path planner for grid-based navigation.
    
    Implements the A* search algorithm to find optimal paths in an occupancy grid.
    """
    
    def __init__(self, occupancy_grid: OccupancyGrid, allow_diagonal: bool = True):
        """
        Initialize the A* path planner.
        
        Args:
            occupancy_grid: OccupancyGrid instance
            allow_diagonal: Whether to allow diagonal movements
        """
        self.grid = occupancy_grid
        self.allow_diagonal = allow_diagonal
        
        # Define movement directions (4-connected or 8-connected)
        if allow_diagonal:
            # 8-connected: includes diagonals
            self.movements = [
                (0, 1),   # up
                (1, 0),   # right
                (0, -1),  # down
                (-1, 0),  # left
                (1, 1),   # up-right
                (1, -1),  # down-right
                (-1, -1), # down-left
                (-1, 1),  # up-left
            ]
            # Cost for diagonal movements is sqrt(2), for cardinal is 1
            self.movement_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        else:
            # 4-connected: only cardinal directions
            self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.movement_costs = [1.0, 1.0, 1.0, 1.0]
    
    def plan(self, 
             start_pos: Tuple[float, float], 
             goal_pos: Tuple[float, float],
             inflation_radius: int = 2) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using A*.
        
        Args:
            start_pos: Start position (x, y) in world coordinates
            goal_pos: Goal position (x, y) in world coordinates
            inflation_radius: Number of cells to inflate obstacles (for safety margin)
            
        Returns:
            List of waypoints (x, y) in world coordinates, or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.grid.world_to_grid(goal_pos[0], goal_pos[1])
        
        # Check if start and goal are valid
        if not self.grid.is_valid_cell(start_grid[0], start_grid[1]):
            print(f"Start position {start_pos} is outside the grid")
            return None
        
        if not self.grid.is_valid_cell(goal_grid[0], goal_grid[1]):
            print(f"Goal position {goal_pos} is outside the grid")
            return None
        
        if not self.grid.is_cell_free(start_grid[0], start_grid[1]):
            print(f"Start position {start_pos} is occupied")
            return None
        
        if not self.grid.is_cell_free(goal_grid[0], goal_grid[1]):
            print(f"Goal position {goal_pos} is occupied")
            return None
        
        # Create inflated occupancy grid for safety
        inflated_grid = self._inflate_obstacles(inflation_radius)
        
        # Run A* search
        path_grid = self._astar_search(start_grid, goal_grid, inflated_grid)
        
        if path_grid is None:
            print(f"No path found from {start_pos} to {goal_pos}")
            return None
        
        # Convert grid path to world coordinates
        path_world = []
        for grid_x, grid_y in path_grid:
            world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)
            path_world.append((world_x, world_y))
        
        # Smooth the path
        path_world = self._smooth_path(path_world)
        
        return path_world
    
    def _astar_search(self, 
                     start: Tuple[int, int], 
                     goal: Tuple[int, int],
                     inflated_grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        A* search algorithm implementation.
        
        Args:
            start: Start grid cell (x, y)
            goal: Goal grid cell (x, y)
            inflated_grid: Binary occupancy grid with inflated obstacles
            
        Returns:
            List of grid cells (x, y) forming the path, or None if no path found
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
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(self.movements):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if neighbor is valid
                if not self.grid.is_valid_cell(neighbor[0], neighbor[1]):
                    continue
                
                # Check if neighbor is occupied (in inflated grid)
                if inflated_grid[neighbor[1], neighbor[0]] > 0:
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
        
        # No path found
        return None
    
    def _heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Heuristic function for A* (Euclidean distance).
        
        Args:
            node: Current node (x, y)
            goal: Goal node (x, y)
            
        Returns:
            Estimated distance to goal
        """
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        
        if self.allow_diagonal:
            # Diagonal distance (Chebyshev)
            return max(dx, dy) + (1.414 - 1) * min(dx, dy)
        else:
            # Manhattan distance
            return dx + dy
    
    def _reconstruct_path(self, 
                         came_from: dict, 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
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
        
        # For each occupied cell, inflate around it
        occupied_cells = np.argwhere(binary_grid > 0)
        
        for y, x in occupied_cells:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Check if within radius (using L-infinity norm for simplicity)
                    if abs(dx) <= radius and abs(dy) <= radius:
                        ny, nx = y + dy, x + dx
                        if self.grid.is_valid_cell(nx, ny):
                            inflated[ny, nx] = 1
        
        return inflated
    
    def _smooth_path(self, path: List[Tuple[float, float]], 
                    window_size: int = 3) -> List[Tuple[float, float]]:
        """
        Smooth the path using a simple moving average filter.
        
        Args:
            path: List of waypoints (x, y)
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
            
            smoothed.append((avg_x, avg_y))
        
        smoothed.append(path[-1])
        
        return smoothed
