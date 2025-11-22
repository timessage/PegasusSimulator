#!/usr/bin/env python
"""
| File: visualizer_3d.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: 3D point cloud visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List

class PointCloudVisualizer3D:
    """
    Visualizer for 3D point clouds using matplotlib.
    """
    
    def __init__(self):
        """Initialize the 3D visualizer."""
        self.fig = None
        self.ax = None
    
    def visualize_point_cloud(self,
                            points: np.ndarray,
                            colors: Optional[np.ndarray] = None,
                            robot_position: Optional[np.ndarray] = None,
                            path: Optional[List[Tuple[float, float, float]]] = None,
                            title: str = "3D Point Cloud Map",
                            point_size: float = 1.0,
                            save_path: Optional[str] = None):
        """
        Visualize a 3D point cloud.
        
        Args:
            points: Nx3 array of 3D points
            colors: Optional Nx3 array of RGB colors [0-1]
            robot_position: Optional robot position [x, y, z]
            path: Optional 3D path to visualize (list of (x, y, z) tuples)
            title: Plot title
            point_size: Size of points in plot
            save_path: If provided, save figure to this path
        """
        if len(points) == 0:
            print("No points to visualize")
            return
        
        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot point cloud
        if colors is not None:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=colors, s=point_size, alpha=0.6)
        else:
            # Color by height if no colors provided
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=points[:, 2], cmap='viridis', s=point_size, alpha=0.6)
            plt.colorbar(self.ax.collections[0], ax=self.ax, label='Height (m)')
        
        # Plot robot position
        if robot_position is not None:
            self.ax.scatter(robot_position[0], robot_position[1], robot_position[2],
                          c='red', s=100, marker='o', label='Robot', 
                          edgecolors='black', linewidths=2)
        
        # Plot path (full 3D path)
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            # Path can be 2D or 3D
            if path_array.shape[1] == 2:
                # 2D path, project to ground plane
                ground_height = np.percentile(points[:, 2], 5) if len(points) > 0 else 0.0
                path_3d = np.column_stack([path_array[:, 0], 
                                          path_array[:, 1], 
                                          np.full(len(path_array), ground_height)])
            else:
                # 3D path, use as is
                path_3d = path_array
            
            self.ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
                       'b-', linewidth=3, label='Planned 3D Path')
            self.ax.scatter(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
                          c='blue', s=50, marker='o', edgecolors='black')
        
        # Set labels and title
        self.ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set equal aspect ratio
        self._set_equal_aspect_3d(points)
        
        # Add legend
        if robot_position is not None or path is not None:
            self.ax.legend(loc='upper right')
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total Points: {len(points):,}\n"
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        stats_text += f"X: [{min_bounds[0]:.1f}, {max_bounds[0]:.1f}] m\n"
        stats_text += f"Y: [{min_bounds[1]:.1f}, {max_bounds[1]:.1f}] m\n"
        stats_text += f"Z: [{min_bounds[2]:.1f}, {max_bounds[2]:.1f}] m"
        
        self.ax.text2D(0.02, 0.98, stats_text, 
                      transform=self.ax.transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_multiple_views(self,
                                points: np.ndarray,
                                colors: Optional[np.ndarray] = None,
                                robot_position: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None):
        """
        Visualize point cloud from multiple viewpoints.
        
        Args:
            points: Nx3 array of 3D points
            colors: Optional Nx3 array of RGB colors [0-1]
            robot_position: Optional robot position [x, y, z]
            save_path: If provided, save figure to this path
        """
        if len(points) == 0:
            print("No points to visualize")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Define viewing angles (elevation, azimuth)
        views = [
            (30, 45, "Isometric View"),
            (0, 0, "Front View (YZ plane)"),
            (0, 90, "Side View (XZ plane)"),
            (90, 0, "Top View (XY plane)")
        ]
        
        for idx, (elev, azim, view_title) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')
            
            # Plot point cloud
            if colors is not None:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=colors, s=1.0, alpha=0.6)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=points[:, 2], cmap='viridis', s=1.0, alpha=0.6)
            
            # Plot robot position
            if robot_position is not None:
                ax.scatter(robot_position[0], robot_position[1], robot_position[2],
                         c='red', s=100, marker='o', 
                         edgecolors='black', linewidths=2)
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Set labels
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(view_title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set equal aspect
            self._set_equal_aspect_3d(points, ax)
        
        plt.suptitle('3D Point Cloud - Multiple Views', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multi-view visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_point_cloud_with_2d_projection(self,
                                                 points: np.ndarray,
                                                 colors: Optional[np.ndarray] = None,
                                                 occupancy_grid: Optional[np.ndarray] = None,
                                                 grid_extent: Optional[Tuple[float, float, float, float]] = None,
                                                 save_path: Optional[str] = None):
        """
        Visualize 3D point cloud alongside 2D occupancy grid projection.
        
        Args:
            points: Nx3 array of 3D points
            colors: Optional Nx3 array of RGB colors [0-1]
            occupancy_grid: Optional 2D occupancy grid for comparison
            grid_extent: Optional extent of grid [x_min, x_max, y_min, y_max]
            save_path: If provided, save figure to this path
        """
        if len(points) == 0:
            print("No points to visualize")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 7))
        
        # 3D view
        ax1 = fig.add_subplot(121, projection='3d')
        
        if colors is not None:
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=colors, s=1.0, alpha=0.6)
        else:
            scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                                c=points[:, 2], cmap='viridis', s=1.0, alpha=0.6)
            plt.colorbar(scatter, ax=ax1, label='Height (m)', pad=0.1)
        
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_zlabel('Z (m)', fontsize=11)
        ax1.set_title('3D Point Cloud', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        self._set_equal_aspect_3d(points, ax1)
        
        # 2D projection
        ax2 = fig.add_subplot(122)
        
        if occupancy_grid is not None:
            # Show occupancy grid
            if grid_extent is None:
                grid_extent = [points[:, 0].min(), points[:, 0].max(),
                              points[:, 1].min(), points[:, 1].max()]
            
            im = ax2.imshow(occupancy_grid, cmap='gray_r', origin='lower',
                          extent=grid_extent, alpha=0.8)
            plt.colorbar(im, ax=ax2, label='Occupancy Probability')
        
        # Overlay point cloud projection
        ax2.scatter(points[:, 0], points[:, 1], 
                   c=points[:, 2], cmap='viridis', s=1.0, alpha=0.5)
        
        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Y (m)', fontsize=11)
        ax2.set_title('2D Projection (Top View)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.suptitle('3D Point Cloud and 2D Occupancy Map', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Combined visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _set_equal_aspect_3d(self, points: np.ndarray, ax=None):
        """
        Set equal aspect ratio for 3D plot.
        
        Args:
            points: Point cloud array
            ax: Matplotlib 3D axis (uses self.ax if None)
        """
        if ax is None:
            ax = self.ax
        
        # Get bounds
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        
        # Calculate ranges
        ranges = max_bounds - min_bounds
        max_range = ranges.max()
        
        # Calculate centers
        centers = (max_bounds + min_bounds) / 2
        
        # Set limits
        ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
        ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
        ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)
    
    def create_animation(self,
                        point_cloud_sequence: List[np.ndarray],
                        colors_sequence: Optional[List[np.ndarray]] = None,
                        save_path: str = "point_cloud_animation",
                        fps: int = 10):
        """
        Create an animated visualization of point cloud evolution by saving frames.
        
        Note: This saves individual frames. To create a video, use external tools like:
        - ffmpeg: ffmpeg -r {fps} -i frame_%04d.png -vcodec libx264 animation.mp4
        - imageio: Requires imageio package for direct video creation
        
        Args:
            point_cloud_sequence: List of point cloud arrays over time
            colors_sequence: Optional list of color arrays for each frame
            save_path: Base path for saving frames (without extension)
            fps: Target frames per second for playback
        """
        if not point_cloud_sequence:
            print("No point cloud data to animate")
            return
        
        print(f"Creating animation with {len(point_cloud_sequence)} frames")
        print(f"Saving frames to {save_path}_frame_*.png")
        
        # Save each frame
        for i, points in enumerate(point_cloud_sequence):
            if len(points) == 0:
                continue
            
            colors = colors_sequence[i] if colors_sequence and i < len(colors_sequence) else None
            frame_path = f"{save_path}_frame_{i:04d}.png"
            
            self.visualize_point_cloud(
                points=points,
                colors=colors,
                title=f"3D Point Cloud Evolution - Frame {i+1}/{len(point_cloud_sequence)}",
                save_path=frame_path
            )
        
        print(f"\nSaved {len(point_cloud_sequence)} frames")
        print(f"To create video with ffmpeg:")
        print(f"  ffmpeg -r {fps} -i {save_path}_frame_%04d.png -vcodec libx264 animation.mp4")
        

def visualize_point_cloud_simple(points: np.ndarray,
                                colors: Optional[np.ndarray] = None,
                                title: str = "3D Point Cloud"):
    """
    Quick function to visualize a point cloud.
    
    Args:
        points: Nx3 array of points
        colors: Optional Nx3 array of RGB colors
        title: Plot title
    """
    viz = PointCloudVisualizer3D()
    viz.visualize_point_cloud(points, colors, title=title)
