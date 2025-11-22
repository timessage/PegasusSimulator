#!/usr/bin/env python
"""
| File: slam_path_planning_3d_example.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: Complete 3D SLAM mapping and path planning example using Pegasus Simulator.
| This example demonstrates:
| 1. Real-time 3D occupancy grid mapping (voxel grid) using lidar data
| 2. 3D point cloud accumulation and visualization
| 3. 3D A* path planning algorithm for aerial navigation
| 4. Autonomous 3D path following with altitude control
| 5. Map and point cloud visualization and saving capabilities
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
import numpy as np
from scipy.spatial.transform import Rotation

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.graphical_sensors.lidar import Lidar

# Import 3D SLAM components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'slam_path_planning'))

from slam_controller_3d import SLAMController3D
from visualizer_3d import PointCloudVisualizer3D

class SLAM3DBackend(Backend):
    """
    Custom backend that integrates 3D SLAM mapping and path planning.
    """
    
    def __init__(self, waypoints: list = None):
        """
        Initialize the 3D SLAM backend.
        
        Args:
            waypoints: List of (x, y, z) goal positions for the drone to visit
        """
        super().__init__()
        
        # Initialize 3D SLAM controller
        self.slam = SLAMController3D(
            map_width=60.0,
            map_height=60.0,
            map_depth=15.0,
            map_resolution=0.3,
            map_origin=(-30.0, -30.0, 0.0)
        )
        
        # Waypoints to visit (3D positions)
        self.waypoints = waypoints if waypoints is not None else [
            (5.0, 5.0, 3.0),
            (10.0, 5.0, 4.0),
            (10.0, 10.0, 5.0),
            (5.0, 10.0, 4.0),
            (0.0, 5.0, 3.0),
            (0.0, 0.0, 2.5)
        ]
        self.current_waypoint_idx = 0
        
        # State tracking
        self.mapping_phase = True
        self.mapping_time = 0.0
        self.mapping_duration = 15.0  # Map for 15 seconds before starting navigation
        
        # Lidar data storage
        self.lidar_data = {'points': None}
        
        # Timing
        self.dt = 0.0
        self.time = 0.0
        
        # Visualization
        self.last_visualization_time = 0.0
        self.visualization_interval = 10.0  # Visualize every 10 seconds
        
        print("="*60)
        print("3D SLAM Path Planning Example Initialized")
        print("="*60)
        print(f"Waypoints to visit:")
        for i, wp in enumerate(self.waypoints, 1):
            print(f"  {i}. {wp}")
        print(f"Mapping duration: {self.mapping_duration}s")
        print("="*60)
    
    def start(self):
        """Called when the simulation starts."""
        print("3D SLAM Backend started - Beginning 3D mapping phase...")
    
    def stop(self):
        """Called when the simulation stops."""
        # Save the 3D map and point cloud
        map_dir = os.path.join(os.path.dirname(__file__), 'slam_path_planning')
        map_base = os.path.join(map_dir, 'saved_map_3d')
        self.slam.save_map(map_base)
        
        # Create final visualization
        print("\nGenerating final 3D visualization...")
        viz_data = self.slam.get_visualization_data()
        
        if len(viz_data['point_cloud_points']) > 0:
            visualizer = PointCloudVisualizer3D()
            
            # Save multiple views
            viz_path = os.path.join(map_dir, 'final_3d_map.png')
            visualizer.visualize_point_cloud(
                points=viz_data['point_cloud_points'],
                colors=viz_data['point_cloud_colors'],
                path=[(p[0], p[1]) for p in self.slam.current_path] if self.slam.current_path else None,
                title="Final 3D SLAM Map",
                save_path=viz_path
            )
            
            # Save multi-view
            multi_view_path = os.path.join(map_dir, 'final_3d_map_multiview.png')
            visualizer.visualize_multiple_views(
                points=viz_data['point_cloud_points'],
                colors=viz_data['point_cloud_colors'],
                save_path=multi_view_path
            )
            
            # Save combined 3D + 2D view
            combined_path = os.path.join(map_dir, 'final_combined_view.png')
            grid_extent = [
                self.slam.occupancy_grid.origin[0],
                self.slam.occupancy_grid.origin[0] + self.slam.occupancy_grid.width,
                self.slam.occupancy_grid.origin[1],
                self.slam.occupancy_grid.origin[1] + self.slam.occupancy_grid.height
            ]
            visualizer.visualize_point_cloud_with_2d_projection(
                points=viz_data['point_cloud_points'],
                colors=viz_data['point_cloud_colors'],
                occupancy_grid=viz_data['occupancy_grid_2d'],
                grid_extent=grid_extent,
                save_path=combined_path
            )
        
        print(f"3D SLAM Backend stopped - Maps and visualizations saved")
    
    def update_sensor(self, sensor_type: str, data):
        """
        Called when sensor data is available.
        
        Args:
            sensor_type: Type of sensor (e.g., "Lidar")
            data: Sensor data
        """
        if sensor_type == "Lidar":
            # Store lidar data
            if data is not None and len(data) > 0:
                # Data is Nx3 array of points in sensor frame
                self.lidar_data['points'] = data
    
    def update(self, dt: float):
        """
        Main update loop called every physics step.
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        self.time += dt
        
        # Get vehicle state
        if not hasattr(self, '_vehicle') or self._vehicle is None:
            return
        
        state = self._vehicle.state
        position = np.array([state.position[0], state.position[1], state.position[2]])
        orientation = state.attitude  # [w, x, y, z] quaternion
        velocity = np.array([state.linear_velocity[0], 
                           state.linear_velocity[1], 
                           state.linear_velocity[2]])
        
        # Transform lidar points to world frame
        lidar_world_points = None
        if self.lidar_data['points'] is not None and len(self.lidar_data['points']) > 0:
            # Get rotation matrix from quaternion
            rot = Rotation.from_quat([orientation[1], orientation[2], 
                                     orientation[3], orientation[0]])
            rot_matrix = rot.as_matrix()
            
            # Transform points: world_point = R * sensor_point + position
            lidar_world_points = (rot_matrix @ self.lidar_data['points'].T).T + position
        
        # Prepare state for SLAM controller
        slam_state = {
            'position': position,
            'orientation': orientation,
            'velocity': velocity,
            'lidar': {'points': lidar_world_points}
        }
        
        # Check if we're in mapping phase
        if self.mapping_phase:
            self.mapping_time += dt
            
            # Periodic status updates
            if int(self.mapping_time) % 3 == 0 and self.mapping_time - dt < int(self.mapping_time):
                pc_stats = self.slam.point_cloud.get_statistics()
                print(f"Mapping... {self.mapping_time:.1f}s / {self.mapping_duration:.1f}s "
                      f"- Points: {pc_stats['num_points']}")
            
            if self.mapping_time >= self.mapping_duration:
                # End mapping phase, start navigation
                self.mapping_phase = False
                print("\n" + "="*60)
                print("3D Mapping phase complete - Starting navigation")
                print("="*60)
                
                # Print map statistics
                map_info = self.slam.occupancy_grid.get_map_info()
                pc_stats = self.slam.point_cloud.get_statistics()
                print(f"Voxel grid statistics:")
                print(f"  - Occupied voxels: {map_info['occupied_voxels']}")
                print(f"  - Free voxels: {map_info['free_voxels']}")
                print(f"  - Unknown voxels: {map_info['unknown_voxels']}")
                print(f"Point cloud statistics:")
                print(f"  - Total points: {pc_stats['num_points']}")
                print(f"  - Scans: {pc_stats['num_scans']}")
                print(f"  - Density: {pc_stats['density']:.2f} points/m³")
                print("="*60)
                
                # Set first waypoint
                if len(self.waypoints) > 0:
                    goal = self.waypoints[self.current_waypoint_idx]
                    current_pos = (position[0], position[1], position[2])
                    self.slam.set_goal(goal, current_pos)
            else:
                # During mapping, just hover and collect data
                self.slam.update_map(position, {'points': lidar_world_points})
                
                # Send hover command at preferred altitude
                reference = [position[0], position[1], 2.5, 0.0]  # x, y, z, yaw
                self.input_reference(reference)
                return
        
        # Navigation phase - update SLAM controller
        control = self.slam.update(slam_state)
        
        # Check if current waypoint reached and update to next
        if self.slam.current_path is None:
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx < len(self.waypoints):
                # Set next waypoint
                goal = self.waypoints[self.current_waypoint_idx]
                current_pos = (position[0], position[1], position[2])
                print(f"\nNavigating to 3D waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}: {goal}")
                
                # Plan path from current position
                if not self.slam.set_goal(goal, current_pos):
                    print(f"Warning: Could not plan 3D path to waypoint {goal}")
            else:
                # All waypoints visited
                print("\n" + "="*60)
                print("All 3D waypoints visited - Mission complete!")
                print("="*60)
                
                # Hover at current position
                reference = [position[0], position[1], position[2], 0.0]
                self.input_reference(reference)
                return
        
        # Periodic visualization during navigation
        if self.time - self.last_visualization_time > self.visualization_interval:
            self.last_visualization_time = self.time
            print(f"\nGenerating intermediate visualization at t={self.time:.1f}s...")
            self._save_visualization()
        
        # Apply control commands
        desired_velocity = control['velocity']
        
        # Calculate target position (integrate velocity)
        target_x = position[0] + desired_velocity[0] * 0.5
        target_y = position[1] + desired_velocity[1] * 0.5
        target_z = position[2] + desired_velocity[2] * 0.5
        
        # Get current yaw and add yaw rate
        rot = Rotation.from_quat([orientation[1], orientation[2], 
                                 orientation[3], orientation[0]])
        current_yaw = rot.as_euler('xyz')[2]
        target_yaw = current_yaw + control['yaw_rate'] * 0.5
        
        # Send reference to vehicle
        reference = [target_x, target_y, target_z, target_yaw]
        self.input_reference(reference)
    
    def _save_visualization(self):
        """Save current 3D visualization."""
        try:
            viz_data = self.slam.get_visualization_data()
            
            if len(viz_data['point_cloud_points']) > 0:
                map_dir = os.path.join(os.path.dirname(__file__), 'slam_path_planning')
                viz_path = os.path.join(map_dir, f'3d_map_t{int(self.time)}.png')
                
                visualizer = PointCloudVisualizer3D()
                visualizer.visualize_point_cloud(
                    points=viz_data['point_cloud_points'],
                    colors=viz_data['point_cloud_colors'],
                    path=[(p[0], p[1]) for p in self.slam.current_path] if self.slam.current_path else None,
                    title=f"3D SLAM Map at t={self.time:.1f}s",
                    save_path=viz_path
                )
                print(f"  Saved visualization to {viz_path}")
        except Exception as e:
            print(f"  Warning: Could not save visualization: {e}")
    
    def input_reference(self, reference: list):
        """
        Send position and yaw reference to the vehicle.
        
        Args:
            reference: [x, y, z, yaw] target position and orientation
        """
        if hasattr(self, '_vehicle') and self._vehicle is not None:
            self._vehicle.position = reference[:3]
            self._vehicle.attitude = reference[3]


class SLAM3DApp:
    """
    Main application for 3D SLAM mapping and path planning demonstration.
    """
    
    def __init__(self):
        """Initialize the 3D SLAM application."""
        
        # Acquire the timeline
        self.timeline = omni.timeline.get_timeline_interface()
        
        # Start the Pegasus Interface
        self.pg = PegasusInterface()
        
        # Initialize the world
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # Load environment with obstacles
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # Define 3D waypoints for the drone to visit
        waypoints = [
            (8.0, 0.0, 3.0),
            (8.0, 8.0, 4.5),
            (0.0, 8.0, 3.5),
            (-8.0, 8.0, 5.0),
            (-8.0, 0.0, 3.0),
            (0.0, 0.0, 2.5)
        ]
        
        # Create the drone with 3D SLAM backend
        config = MultirotorConfig()
        config.backends = [SLAM3DBackend(waypoints=waypoints)]
        
        # Add lidar sensor with good coverage
        config.graphical_sensors = [
            Lidar("lidar", config={
                "position": np.array([0.0, 0.0, 0.05]),
                "orientation": np.array([0.0, 0.0, 0.0]),
                "sensor_configuration": "Example_Rotary",
                "frequency": 20.0
            })
        ]
        
        # Spawn the drone
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config
        )
        
        # Reset the simulation
        self.world.reset()
        
        self.stop_sim = False
        
        print("\n" + "="*60)
        print("3D SLAM Path Planning Example")
        print("="*60)
        print("This example demonstrates:")
        print("1. Real-time 3D SLAM mapping using lidar data")
        print("2. 3D point cloud accumulation")
        print("3. 3D A* path planning for aerial navigation")
        print("4. 3D path following with altitude control")
        print("="*60)
        print("\nSimulation starting...")
        print("="*60 + "\n")
    
    def run(self):
        """Run the simulation loop."""
        
        # Start the simulation
        self.timeline.play()
        
        # Main loop
        while simulation_app.is_running() and not self.stop_sim:
            # Step the simulation
            self.world.step(render=True)
            
            # Check if simulation stopped
            if self.world.is_stopped():
                self.stop_sim = True
        
        # Cleanup
        print("\n3D SLAM Path Planning Example closing...")
        self.timeline.stop()
        simulation_app.close()


def main():
    """Main entry point."""
    
    # Create and run the application
    app = SLAM3DApp()
    app.run()


if __name__ == "__main__":
    main()
