#!/usr/bin/env python
"""
| File: slam_path_planning_example.py
| Author: SLAM Path Planning Example
| License: BSD-3-Clause. Copyright (c) 2024. All rights reserved.
| Description: Complete SLAM mapping and path planning example using Pegasus Simulator.
| This example demonstrates:
| 1. Real-time occupancy grid mapping using lidar data
| 2. A* path planning algorithm for navigation
| 3. Autonomous path following with obstacle avoidance
| 4. Map visualization and saving capabilities
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

# Import SLAM components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'slam_path_planning'))

from slam_controller import SLAMController

class SLAMBackend(Backend):
    """
    Custom backend that integrates SLAM mapping and path planning.
    """
    
    def __init__(self, waypoints: list = None):
        """
        Initialize the SLAM backend.
        
        Args:
            waypoints: List of (x, y) goal positions for the drone to visit
        """
        super().__init__()
        
        # Initialize SLAM controller
        self.slam = SLAMController(
            map_width=60.0,
            map_height=60.0,
            map_resolution=0.2,
            map_origin=(-30.0, -30.0)
        )
        
        # Waypoints to visit
        self.waypoints = waypoints if waypoints is not None else [
            (5.0, 5.0),
            (10.0, 5.0),
            (10.0, 10.0),
            (5.0, 10.0),
            (0.0, 0.0)
        ]
        self.current_waypoint_idx = 0
        
        # State tracking
        self.mapping_phase = True
        self.mapping_time = 0.0
        self.mapping_duration = 10.0  # Map for 10 seconds before starting navigation
        
        # Lidar data storage
        self.lidar_data = {'points': None}
        
        # Timing
        self.dt = 0.0
        self.time = 0.0
        
        print("="*60)
        print("SLAM Path Planning Example Initialized")
        print("="*60)
        print(f"Waypoints to visit: {self.waypoints}")
        print(f"Mapping duration: {self.mapping_duration}s")
        print("="*60)
    
    def start(self):
        """Called when the simulation starts."""
        print("SLAM Backend started - Beginning mapping phase...")
    
    def stop(self):
        """Called when the simulation stops."""
        # Save the map
        map_file = os.path.join(os.path.dirname(__file__), 'slam_path_planning', 'saved_map.npz')
        self.slam.save_map(map_file)
        print(f"SLAM Backend stopped - Map saved to {map_file}")
    
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
                # We need to transform to world frame
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
            
            if self.mapping_time >= self.mapping_duration:
                # End mapping phase, start navigation
                self.mapping_phase = False
                print("\n" + "="*60)
                print("Mapping phase complete - Starting navigation")
                print("="*60)
                
                # Print map statistics
                map_info = self.slam.occupancy_grid.get_map_info()
                print(f"Map statistics:")
                print(f"  - Occupied cells: {map_info['occupied_cells']}")
                print(f"  - Free cells: {map_info['free_cells']}")
                print(f"  - Unknown cells: {map_info['unknown_cells']}")
                print("="*60)
                
                # Set first waypoint
                if len(self.waypoints) > 0:
                    goal = self.waypoints[self.current_waypoint_idx]
                    self.slam.set_goal(goal)
            else:
                # During mapping, just hover and collect data
                self.slam.update_map(position, {'points': lidar_world_points})
                
                # Send hover command
                reference = [position[0], position[1], 2.0, 0.0]  # x, y, z, yaw
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
                print(f"\nNavigating to waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}: {goal}")
                
                # Update path from current position
                current_pos = (position[0], position[1])
                path = self.slam.path_planner.plan(
                    start_pos=current_pos,
                    goal_pos=goal,
                    inflation_radius=3
                )
                
                if path is not None:
                    self.slam.current_path = path
                    self.slam.current_waypoint_idx = 0
                else:
                    print(f"Warning: Could not plan path to waypoint {goal}")
            else:
                # All waypoints visited
                print("\n" + "="*60)
                print("All waypoints visited - Mission complete!")
                print("="*60)
                
                # Hover at current position
                reference = [position[0], position[1], 2.0, 0.0]
                self.input_reference(reference)
                return
        
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
    
    def input_reference(self, reference: list):
        """
        Send position and yaw reference to the vehicle.
        
        Args:
            reference: [x, y, z, yaw] target position and orientation
        """
        if hasattr(self, '_vehicle') and self._vehicle is not None:
            self._vehicle.position = reference[:3]
            self._vehicle.attitude = reference[3]


class SLAMApp:
    """
    Main application for SLAM mapping and path planning demonstration.
    """
    
    def __init__(self):
        """Initialize the SLAM application."""
        
        # Acquire the timeline
        self.timeline = omni.timeline.get_timeline_interface()
        
        # Start the Pegasus Interface
        self.pg = PegasusInterface()
        
        # Initialize the world
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # Load environment with obstacles
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # Define waypoints for the drone to visit
        waypoints = [
            (8.0, 0.0),
            (8.0, 8.0),
            (0.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 0.0),
            (0.0, 0.0)
        ]
        
        # Create the drone with SLAM backend
        config = MultirotorConfig()
        config.backends = [SLAMBackend(waypoints=waypoints)]
        
        # Add lidar sensor
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
        print("SLAM Path Planning Example")
        print("="*60)
        print("This example demonstrates:")
        print("1. Real-time SLAM mapping using lidar data")
        print("2. A* path planning for autonomous navigation")
        print("3. Path following with obstacle avoidance")
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
        print("\nSLAM Path Planning Example closing...")
        self.timeline.stop()
        simulation_app.close()


def main():
    """Main entry point."""
    
    # Create and run the application
    app = SLAMApp()
    app.run()


if __name__ == "__main__":
    main()
