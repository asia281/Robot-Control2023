import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)
viewer.cam.distance = 4.
viewer.cam.lookat = np.array([0, 0, 1])
viewer.cam.elevation = -30.

from drone_simulator import DroneSimulator
from pid import PID

if __name__ == '__main__':
    desired_altitude = 2

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, viewer, desired_altitude = desired_altitude,
        altitude_sensor_freq = 0.01, wind_change_prob = 0.1, rendering_freq = 1
        )

    # TODO: Create necessary PID controllers using PID class
    outer_loop_pid = PID(
        gain_prop=0.1,
        gain_int=0.01,
        gain_der=0.001,
        sensor_period=drone_simulator.altitude_sensor_period
    )

    inner_loop_pid = PID(
        gain_prop=0.1,
        gain_int=0.01,
        gain_der=0.001,
        sensor_period=1 / drone_simulator.sensor_freq_acceleration  # Invert frequency to get period
    )

    # Increase the number of iterations for a longer simulation
    for i in range(4000):
        # TODO: Use the PID controllers in a cascade designe to control the drone
        acceleration_reading = drone_simulator.measured_acceleration[2]  # Only interested in z-direction

        # Outer loop control: Control acceleration based on altitude
        desired_acceleration = outer_loop_pid.output_signal(desired_altitude, drone_simulator.measured_altitudes)
        
        # Inner loop control: Control thrust based on acceleration
        desired_thrust = inner_loop_pid.output_signal(desired_acceleration, acceleration_reading)
        
        drone_simulator.sim_step(desired_thrust)