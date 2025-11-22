#!/usr/bin/env python
"""
| File: point_cloud_3d.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: 3D point cloud storage and processing for SLAM
"""

import numpy as np
from typing import Tuple, List, Optional

class PointCloud3D:
    """
    3D point cloud storage and processing for SLAM applications.
    
    This class manages a collection of 3D points from lidar scans,
    supporting incremental updates, voxel filtering, and visualization.
    """
    
    def __init__(self, 
                 voxel_size: float = 0.1,
                 max_points: int = 1000000):
        """
        Initialize the 3D point cloud.
        
        Args:
            voxel_size: Size of voxel for downsampling (meters)
            max_points: Maximum number of points to store
        """
        self.voxel_size = voxel_size
        self.max_points = max_points
        
        # Point cloud storage
        self.points = np.zeros((0, 3), dtype=np.float32)  # Nx3 array of points
        self.colors = np.zeros((0, 3), dtype=np.float32)  # Nx3 array of RGB colors
        
        # Voxel grid for efficient filtering
        self.voxel_grid = {}  # Dictionary mapping voxel coordinates to point indices
        
        # Statistics
        self.total_points_added = 0
        self.num_scans = 0
        
    def add_scan(self, 
                points: np.ndarray,
                colors: Optional[np.ndarray] = None,
                robot_position: Optional[np.ndarray] = None):
        """
        Add a new lidar scan to the point cloud.
        
        Args:
            points: Nx3 array of 3D points in world coordinates
            colors: Optional Nx3 array of RGB colors [0-1]
            robot_position: Optional robot position for filtering
        """
        if points is None or len(points) == 0:
            return
        
        # Ensure points is 2D array
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Filter invalid points (NaN, inf)
        valid_mask = np.all(np.isfinite(points), axis=1)
        points = points[valid_mask]
        
        if len(points) == 0:
            return
        
        # Generate default colors if not provided (height-based coloring)
        if colors is None:
            # Color by height (z coordinate)
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            if z_max > z_min:
                normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
            else:
                normalized_z = np.zeros(len(points))
            
            # Create colormap: blue (low) -> green (mid) -> red (high)
            colors = np.zeros((len(points), 3))
            colors[:, 0] = normalized_z  # Red channel
            colors[:, 1] = 1.0 - np.abs(normalized_z - 0.5) * 2  # Green channel
            colors[:, 2] = 1.0 - normalized_z  # Blue channel
        else:
            colors = colors[valid_mask]
        
        # Apply voxel grid filtering to new points
        filtered_points = []
        filtered_colors = []
        
        for i, point in enumerate(points):
            voxel_key = self._get_voxel_key(point)
            
            # Only add point if this voxel is empty or we're replacing
            if voxel_key not in self.voxel_grid:
                filtered_points.append(point)
                filtered_colors.append(colors[i])
                self.voxel_grid[voxel_key] = len(self.points) + len(filtered_points) - 1
        
        if len(filtered_points) == 0:
            return
        
        # Convert to arrays
        filtered_points = np.array(filtered_points, dtype=np.float32)
        filtered_colors = np.array(filtered_colors, dtype=np.float32)
        
        # Add to point cloud
        self.points = np.vstack([self.points, filtered_points])
        self.colors = np.vstack([self.colors, filtered_colors])
        
        # Update statistics
        self.total_points_added += len(points)
        self.num_scans += 1
        
        # Limit total number of points
        if len(self.points) > self.max_points:
            self._downsample()
    
    def _get_voxel_key(self, point: np.ndarray) -> Tuple[int, int, int]:
        """
        Get voxel grid key for a point.
        
        Args:
            point: 3D point [x, y, z]
            
        Returns:
            Voxel key (ix, iy, iz)
        """
        voxel_x = int(np.floor(point[0] / self.voxel_size))
        voxel_y = int(np.floor(point[1] / self.voxel_size))
        voxel_z = int(np.floor(point[2] / self.voxel_size))
        return (voxel_x, voxel_y, voxel_z)
    
    def _downsample(self):
        """
        Downsample point cloud when it exceeds maximum size.
        Uses random sampling to keep the most recent points.
        """
        if len(self.points) <= self.max_points:
            return
        
        # Keep most recent points (last max_points points)
        keep_indices = np.arange(len(self.points) - self.max_points, len(self.points))
        
        self.points = self.points[keep_indices]
        self.colors = self.colors[keep_indices]
        
        # Rebuild voxel grid
        self.voxel_grid = {}
        for i, point in enumerate(self.points):
            voxel_key = self._get_voxel_key(point)
            self.voxel_grid[voxel_key] = i
    
    def get_points(self) -> np.ndarray:
        """
        Get all points in the cloud.
        
        Returns:
            Nx3 array of points
        """
        return self.points.copy()
    
    def get_colors(self) -> np.ndarray:
        """
        Get colors for all points.
        
        Returns:
            Nx3 array of RGB colors
        """
        return self.colors.copy()
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of the point cloud.
        
        Returns:
            (min_bounds, max_bounds) where each is [x, y, z]
        """
        if len(self.points) == 0:
            return np.zeros(3), np.zeros(3)
        
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        return min_bounds, max_bounds
    
    def filter_by_height(self, z_min: float = -np.inf, z_max: float = np.inf) -> np.ndarray:
        """
        Filter points by height (z coordinate).
        
        Args:
            z_min: Minimum height
            z_max: Maximum height
            
        Returns:
            Indices of points within height range
        """
        mask = (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max)
        return np.where(mask)[0]
    
    def filter_by_distance(self, 
                          center: np.ndarray, 
                          max_distance: float) -> np.ndarray:
        """
        Filter points by distance from a center point.
        
        Args:
            center: Center point [x, y, z]
            max_distance: Maximum distance
            
        Returns:
            Indices of points within distance
        """
        distances = np.linalg.norm(self.points - center, axis=1)
        return np.where(distances <= max_distance)[0]
    
    def project_to_2d(self, plane: str = 'xy') -> np.ndarray:
        """
        Project point cloud to 2D plane.
        
        Args:
            plane: Plane to project to ('xy', 'xz', or 'yz')
            
        Returns:
            Nx2 array of 2D points
        """
        if plane == 'xy':
            return self.points[:, :2]
        elif plane == 'xz':
            return self.points[:, [0, 2]]
        elif plane == 'yz':
            return self.points[:, [1, 2]]
        else:
            raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")
    
    def get_statistics(self) -> dict:
        """
        Get point cloud statistics.
        
        Returns:
            Dictionary with statistics
        """
        if len(self.points) == 0:
            return {
                'num_points': 0,
                'num_scans': self.num_scans,
                'total_points_added': self.total_points_added,
                'bounds': (np.zeros(3), np.zeros(3)),
                'density': 0.0
            }
        
        min_bounds, max_bounds = self.get_bounds()
        volume = np.prod(max_bounds - min_bounds)
        density = len(self.points) / volume if volume > 0 else 0
        
        return {
            'num_points': len(self.points),
            'num_scans': self.num_scans,
            'total_points_added': self.total_points_added,
            'bounds': (min_bounds, max_bounds),
            'density': density,
            'voxel_size': self.voxel_size
        }
    
    def clear(self):
        """Clear all points from the cloud."""
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.voxel_grid = {}
        self.num_scans = 0
    
    def save(self, filename: str):
        """
        Save point cloud to file.
        
        Args:
            filename: Output filename (.npz format)
        """
        np.savez(filename,
                points=self.points,
                colors=self.colors,
                voxel_size=self.voxel_size,
                num_scans=self.num_scans,
                total_points_added=self.total_points_added)
        print(f"Point cloud saved to {filename} ({len(self.points)} points)")
    
    @classmethod
    def load(cls, filename: str) -> 'PointCloud3D':
        """
        Load point cloud from file.
        
        Args:
            filename: Input filename (.npz format)
            
        Returns:
            PointCloud3D instance
        """
        data = np.load(filename)
        
        pc = cls(voxel_size=float(data['voxel_size']))
        pc.points = data['points']
        pc.colors = data['colors']
        pc.num_scans = int(data['num_scans'])
        pc.total_points_added = int(data['total_points_added'])
        
        # Rebuild voxel grid
        for i, point in enumerate(pc.points):
            voxel_key = pc._get_voxel_key(point)
            pc.voxel_grid[voxel_key] = i
        
        print(f"Point cloud loaded from {filename} ({len(pc.points)} points)")
        return pc
    
    def export_ply(self, filename: str):
        """
        Export point cloud to PLY format for external visualization.
        
        Args:
            filename: Output filename (.ply format)
        """
        if len(self.points) == 0:
            print("No points to export")
            return
        
        # Convert colors to 0-255 range
        colors_255 = (self.colors * 255).astype(np.uint8)
        
        # Write PLY file
        with open(filename, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Data
            for point, color in zip(self.points, colors_255):
                f.write(f"{point[0]} {point[1]} {point[2]} "
                       f"{color[0]} {color[1]} {color[2]}\n")
        
        print(f"Point cloud exported to {filename} ({len(self.points)} points)")
