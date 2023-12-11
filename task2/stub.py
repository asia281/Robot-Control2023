"""
Stub for homework 2
"""
import time
import random
import numpy as np
import mujoco
from mujoco import viewer
from PIL import Image


model = mujoco.MjModel.from_xml_path("lab7-public/car.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(forward, turn, steps=1000, view=False):
    data.actuator("forward").ctrl = forward
    data.actuator("turn").ctrl = turn
    for _ in range(steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / 10)

    renderer.update_scene(data, camera="camera1")
    img = renderer.render()
    return img


def task_1_step(turn):
    return sim_step(0.1, turn, steps=200, view=True)

# Function to get the distance between car and ball
def get_distance_ball_car():
    return np.linalg.norm(get_direction_vector())

    # Function to get the direction vector from car to ball
def get_direction_vector():
    car_pos = data.body_xpos[model.body_name2id('car')]
    ball_pos = data.body_xpos[model.body_name2id('ball')]
    direction_vector = ball_pos - car_pos
    return direction_vector

# Function to calculate the angle between two vectors
def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / magnitude_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def task_1():
    steps = random.randint(0, 2000)
    img = sim_step(0, 0.1, steps, view=False)

    # TODO: change the lines below,
    # for car control, you should use task_1_step(turn) function
    for i in range(100):
        print(data.body("car").xpos)
        print(data.body("target-ball").xpos)

        direction_vector = get_direction_vector()

        # Calculate the angle between the car's direction and the direction to the ball
        angle_to_destination = angle_between(data.body_xquat[model.body_name2id('car')], direction_vector)


        # Ensure the car turns smoothly towards the ball
        # if direction_vector[1] < 0:
        #     angle_to_turn *= -1

        turn = angle_to_destination * 2
        task_1_step(turn)
        #sim_step(0.1, turn, view=True)
        distance_to_destination = get_distance_ball_car()
        time.sleep(model.opt.timestep)

    distance = get_distance_ball_car()
    print("Distance between Car and Red Ball:", distance)
    # at the end, your car should be close to the red ball (0.2 distance is fine)
    # data.body("car").xpos) is the position of the car

def get_distance(body1, body2):
    pos1 = data.body_xpos[model.body_name2id(body1)]
    pos2 = data.body_xpos[model.body_name2id(body2)]
    return np.linalg.norm(pos1 - pos2)

def task_2():
    sim_step(0.5, 0, 1000, view=True)
    speed = random.uniform(0.3, 0.5)
    turn = random.uniform(-0.2, 0.2)
    ball_name = 'ball'
    box_name = 'box'
    img = sim_step(speed, turn, 1000, view=True)
    # TODO: change the lines below,
    # you should use sim_step(forward, turn) function
    # you can change the speed and turn as you want
    # do not change the number of steps (1000)
    for step in range(1000):
        # Drive the car forward and randomly turn
        speed = random.uniform(0.3, 0.5)
        turn = random.uniform(-0.2, 0.2)
        sim_step(speed, turn, view=False)

        # Get the distance between the ball and the box
        distance_ball_to_box = get_distance(ball_name, box_name)

        # If the ball is close to the box and the box has not moved too much, break the loop
        if distance_ball_to_box < 0.25 and get_distance('box', 'box_initial') < 0.1:
            print("Ball is close to the box without moving the box too much!")
            break

    # at the end, red ball should be close to the green box (0.25 distance is fine)


drift = 0


def task3_step(forward, turn, steps=1000, view=False):
    sim_step(forward, turn + drift, steps=steps, view=view)


def task_3():
    global drift
    drift = np.random.uniform(-0.1, 0.1)
    # TODO: change the lines below,
    # you should use task3_step(forward, turn, steps) function

    # at the end, car should be between the two boxes
