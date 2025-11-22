#!/usr/bin/env python
"""
| File: visualizer.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: Visualization utilities for SLAM maps and paths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Optional

class MapVisualizer:
    """
    Visualizer for SLAM maps and planned paths.
    """
    
    def __init__(self, occupancy_grid):
        """
        Initialize the map visualizer.
        
        Args:
            occupancy_grid: OccupancyGrid instance to visualize
        """
        self.grid = occupancy_grid
        self.fig = None
        self.ax = None
        
    def visualize_map(self, 
                     path: Optional[List[Tuple[float, float]]] = None,
                     robot_pos: Optional[Tuple[float, float]] = None,
                     goal_pos: Optional[Tuple[float, float]] = None,
                     save_path: Optional[str] = None):
        """
        Visualize the occupancy grid map with optional path and positions.
        
        Args:
            path: List of waypoints (x, y) in world coordinates
            robot_pos: Current robot position (x, y)
            goal_pos: Goal position (x, y)
            save_path: If provided, save figure to this path
        """
        # Get probability grid
        prob_grid = self.grid.get_probability_grid()
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Display the occupancy grid
        # Color map: white = free, black = occupied, gray = unknown
        extent = [
            self.grid.origin[0],
            self.grid.origin[0] + self.grid.width,
            self.grid.origin[1],
            self.grid.origin[1] + self.grid.height
        ]
        
        im = self.ax.imshow(
            prob_grid, 
            cmap='gray_r',
            origin='lower',
            extent=extent,
            vmin=0,
            vmax=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax)
        cbar.set_label('Occupancy Probability', rotation=270, labelpad=20)
        
        # Plot path if provided
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            self.ax.plot(path_array[:, 0], path_array[:, 1], 
                        'b-', linewidth=2, label='Planned Path')
            self.ax.plot(path_array[:, 0], path_array[:, 1], 
                        'bo', markersize=4)
        
        # Plot robot position
        if robot_pos is not None:
            self.ax.plot(robot_pos[0], robot_pos[1], 
                        'go', markersize=15, label='Robot')
            self.ax.add_patch(Circle(robot_pos, 0.3, 
                                    color='green', fill=False, linewidth=2))
        
        # Plot goal position
        if goal_pos is not None:
            self.ax.plot(goal_pos[0], goal_pos[1], 
                        'r*', markersize=20, label='Goal')
            self.ax.add_patch(Circle(goal_pos, 0.3, 
                                    color='red', fill=False, linewidth=2))
        
        # Set labels and title
        self.ax.set_xlabel('X (meters)', fontsize=12)
        self.ax.set_ylabel('Y (meters)', fontsize=12)
        self.ax.set_title('SLAM Occupancy Grid Map', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Map visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_exploration_progress(self,
                                      exploration_history: List[dict],
                                      save_path: Optional[str] = None):
        """
        Visualize the exploration progress over time.
        
        Args:
            exploration_history: List of dictionaries with map statistics over time
            save_path: If provided, save figure to this path
        """
        if not exploration_history:
            print("No exploration history to visualize")
            return
        
        # Extract statistics
        times = [i for i in range(len(exploration_history))]
        occupied = [h.get('occupied_cells', 0) for h in exploration_history]
        free = [h.get('free_cells', 0) for h in exploration_history]
        unknown = [h.get('unknown_cells', 0) for h in exploration_history]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot cell counts
        ax1.plot(times, occupied, 'r-', linewidth=2, label='Occupied')
        ax1.plot(times, free, 'g-', linewidth=2, label='Free')
        ax1.plot(times, unknown, 'gray', linewidth=2, label='Unknown')
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Number of Cells', fontsize=12)
        ax1.set_title('Map Exploration Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot exploration percentage
        total_cells = self.grid.grid_width * self.grid.grid_height
        explored = [((o + f) / total_cells * 100) for o, f in zip(occupied, free)]
        
        ax2.plot(times, explored, 'b-', linewidth=2)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Explored Area (%)', fontsize=12)
        ax2.set_title('Exploration Coverage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Exploration progress saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def load_and_visualize_saved_map(map_file: str, save_path: Optional[str] = None):
        """
        Load and visualize a saved map.
        
        Args:
            map_file: Path to saved map file (.npz format)
            save_path: If provided, save figure to this path
        """
        # Load map data
        data = np.load(map_file)
        
        # Extract map information
        grid = data['grid']
        width = data['width']
        height = data['height']
        resolution = data['resolution']
        origin = data['origin']
        
        # Convert log-odds to probability
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display the occupancy grid
        extent = [
            origin[0],
            origin[0] + width,
            origin[1],
            origin[1] + height
        ]
        
        im = ax.imshow(
            prob_grid, 
            cmap='gray_r',
            origin='lower',
            extent=extent,
            vmin=0,
            vmax=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Occupancy Probability', rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'Saved SLAM Map\n(Resolution: {resolution}m, Size: {width}x{height}m)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved map visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_quick_visualization(occupancy_grid, 
                               path=None, 
                               robot_pos=None, 
                               goal_pos=None,
                               title="SLAM Map"):
    """
    Quick function to create a simple visualization.
    
    Args:
        occupancy_grid: OccupancyGrid instance
        path: Optional path to visualize
        robot_pos: Optional robot position
        goal_pos: Optional goal position
        title: Plot title
    """
    visualizer = MapVisualizer(occupancy_grid)
    visualizer.visualize_map(path, robot_pos, goal_pos)
