import os
import carla

import random
import queue

import cv2
import torch
import numpy as np

## Part 0: Object Detection model

from what.models.detection.frcnn.faster_rcnn import FasterRCNN
# from what.models.detection.frcnn.datasets.util import read_image

from what.cli.model import *
from what.utils.file import get_file

from what.models.detection.utils.box_utils import draw_bounding_boxes
from what.models.detection.datasets.voc import VOC_CLASS_NAMES

# Check what_model_list for all available models
index = 8

# Download the model first if not exists
WHAT_MODEL_FILE = what_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_MODEL_URL  = what_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_MODEL_HASH = what_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE)):
    get_file(WHAT_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_MODEL_URL,
             WHAT_MODEL_HASH)

# Faster RCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FasterRCNN(device=device)
model.load(os.path.join(WHAT_MODEL_PATH, WHAT_MODEL_FILE), map_location=device)

## Part 1: CARLA Initialization

# Connect to Carla
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get a vehicle from the library
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')

# Get a spawn point
spawn_points = world.get_map().get_spawn_points()

# Spawn a vehicle
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Spawn NPC
for i in range(20):
    vehicle_bp = bp_lib.filter('vehicle')

    # Exclude bicycle
    car_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn_points))

    if npc:
        npc.set_autopilot(True)

# Get the world spectator 
spectator = world.get_spectator() 

## Part 2: Camera Callback

# Create a camera floating behind the vehicle
camera_init_trans = carla.Transform(carla.Location(x=1, z=2))

# Create a RGB camera
rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# [Windows Only] Fixes https://github.com/carla-simulator/carla/issues/6085
rgb_camera_bp.set_attribute('image_size_x', '640')
rgb_camera_bp.set_attribute('image_size_y', '640')

camera = world.spawn_actor(rgb_camera_bp, camera_init_trans, attach_to=vehicle)

# Callback stores sensor data in a dictionary for use outside callback                         
def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))

# Get gamera dimensions and initialise dictionary                       
image_w = rgb_camera_bp.get_attribute("image_size_x").as_int()
image_h = rgb_camera_bp.get_attribute("image_size_y").as_int()

# Start camera recording
rgb_image_queue = queue.Queue()
camera.listen(lambda image: camera_callback(image, rgb_image_queue))

# Clear the spawned vehicle and camera
def clear():
    settings = world.get_settings()
    settings.synchronous_mode = False # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()

    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    print("Vehicles Destroyed.")

# Autopilot
vehicle.set_autopilot(True) 

# Main loop
while True:
    try:
        world.tick()

        # Move the spectator to the top of the vehicle 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=50)), carla.Rotation(yaw=-180, pitch=-90)) 
        spectator.set_transform(transform) 

        # Display RGB camera image
        origin_image =  rgb_image_queue.get()

        # Image preprocessing
        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

        # Image preprocessing
        height, width, _ = image.shape

        # NHWC -> NCHW
        input = np.array(image).transpose((2, 0, 1))
        input = torch.from_numpy(input)[None]

        inputs, boxes, labels, scores = model.predict(input)

        # (x1, y1, x2, y2) --> (c1, c2, w, h) (0.0, 1.0)
        boxes = np.array(boxes)[0]
        box_w  = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] += box_w / 2
        boxes[:, 0] /= width
        boxes[:, 1] += box_h / 2
        boxes[:, 1] /= height
        boxes[:, 2] = box_w / width
        boxes[:, 3] = box_h / height

        # Only draw 6: bus, 7: car (remove background)
        boxes = np.array([box for box, label in zip(boxes, labels[0]) if label in [6, 7]])
        scores = np.array([score for score, label in zip(scores[0], labels[0]) if label in [6, 7]])
        labels = np.array([6 for label in labels[0] if label in [6, 7]])

        output = draw_bounding_boxes(origin_image,
                boxes,
                labels,
                VOC_CLASS_NAMES[1:],
                scores)

        cv2.imshow('2D Faster RCNN', output)

        # Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            clear()
            break

    except KeyboardInterrupt as e:
        clear()
        break
