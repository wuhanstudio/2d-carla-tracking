import os

from sort.sort import Sort

import carla

import random
import queue

import cv2
import torch
import numpy as np

## Part 0: Object Detection model

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.yolo.yolov4 import YOLOV4
from what.models.detection.yolo.yolov4_tiny import YOLOV4_TINY

from what.cli.model import *
from what.utils.file import get_file

def draw_bounding_boxes(image, boxes, labels, class_names, ids):
    if not hasattr(draw_bounding_boxes, "colours"):
        draw_bounding_boxes.colours = np.random.randint(0, 256, size=(32, 3))

    if len(boxes) > 0:
        assert(boxes.shape[1] == 4)

    # (x, y, w, h) --> (x1, y1, x2, y2)
    height, width, _ = image.shape
    for box in boxes:
        box[0] *= width
        box[1] *= height
        box[2] *= width 
        box[3] *= height

        # From center to top left
        box[0] -= box[2] / 2
        box[1] -= box[3] / 2

        # From width and height to x2 and y2
        box[2] += box[0]
        box[3] += box[1]

    # Draw bounding boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = f"{class_names[labels[i]]}: {int(ids[i])}"
        # print(label)

        # Draw bounding boxes
        cv2.rectangle(  image, 
                        (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), 
                        tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]), 
                        4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]),
                    2)  # line type
    return image

# Check what_model_list for all supported models
what_yolov4_model_list = what_model_list[4:6]

index = 0 # YOLOv4
# index = 1 # YOLOv4 Tiny

# Download the model first if not exists
WHAT_YOLOV4_MODEL_FILE = what_yolov4_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLOV4_MODEL_URL  = what_yolov4_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLOV4_MODEL_HASH = what_yolov4_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE)):
    get_file(WHAT_YOLOV4_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLOV4_MODEL_URL,
             WHAT_YOLOV4_MODEL_HASH)

# Darknet
model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))
# model = YOLOV4_TINY(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))

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

mot_tracker = Sort( max_age=1, 
                    min_hits=3,
                    iou_threshold=0.3) #create instance of the SORT tracker

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

        # Run inference
        images, boxes, labels, probs = model.predict(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Only draw 2: car, 5: bus, 7: truck
        boxes = np.array([box for box, label in zip(boxes, labels) if label in [2, 5, 7]])
        probs = np.array([prob for prob, label in zip(probs, labels) if label in [2, 5, 7]])
        labels = np.array([2 for label in labels if label in [2, 5, 7]])

        # convert [x1, y1, w, h] to [x1, y1, x2, y2]
        if len(boxes) != 0:
            sort_boxes = boxes.copy()

            # (xc, yc, w, h) --> (x1, y1, x2, y2)
            height, width, _ = image.shape

            for box in sort_boxes:
                box[0] *= width
                box[1] *= height
                box[2] *= width 
                box[3] *= height

                # From center to top left
                box[0] -= box[2] / 2
                box[1] -= box[3] / 2

                # From width and height to x2 and y2
                box[2] += box[0]
                box[3] += box[1]

            dets = np.concatenate((sort_boxes, probs.reshape((len(probs), -1))), axis=1)

            # Update tracker
            trackers = mot_tracker.update(dets)

            # convert [x1, y1, x2, y2] to [x, y, w, h ]
            for track in trackers:
                # From x2 and y2 to width and height
                track[2] -= track[0]
                track[3] -= track[1]

                # From top left to center
                track[0] += track[2] / 2
                track[1] += track[3] / 2

                track[0] /= width
                track[1] /= height
                track[2] /= width 
                track[3] /= height

            # Draw bounding boxes onto the image
            output = draw_bounding_boxes(image, trackers[:, 0:4], labels, model.class_names, trackers[:, 4]);

        cv2.imshow('2D YOLOv4 SORT', image)

        # Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            clear()
            break

    except KeyboardInterrupt as e:
        clear()
        break
