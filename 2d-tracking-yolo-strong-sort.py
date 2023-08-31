import os
import carla

import queue
import random

import cv2
import numpy as np
import tensorflow as tf

## Part 0: Object Detection model

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.yolo.yolov4 import YOLOV4
from what.models.detection.yolo.yolov4_tiny import YOLOV4_TINY

from what.cli.model import *
from what.utils.file import get_file

from strong_sort import nn_matching
from strong_sort.detection import Detection
from strong_sort.tracker import Tracker

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.compat.v1.import_graph_def(graph_def, name="net")
        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

def draw_bounding_boxes(image, boxes, labels, class_names, ids):
    if not hasattr(draw_bounding_boxes, "colours"):
        draw_bounding_boxes.colours = np.random.randint(0, 256, size=(32, 3))

    if len(boxes) > 0:
        boxes = np.array(boxes)
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

# Deep SORT
encoder = create_box_encoder("mars-small128.pb", batch_size=32)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, None)
tracker = Tracker(metric)

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
        # labels = np.array([2 for label in labels if label in [2, 5, 7]])

        detections = []

        height, width, _ = image.shape

        # Convert [xc, yc, w, h] to [x1, y1, w, h]
        if len(boxes) != 0:
            sort_boxes = boxes.copy()

            # (xc, yc, w, h) --> (x1, y1, x2, y2)
            for i, box in enumerate(sort_boxes):
                box[0] *= width
                box[1] *= height
                box[2] *= width 
                box[3] *= height

                # From center to top left
                box[0] -= box[2] / 2
                box[1] -= box[3] / 2

                # From width and height to x2 and y2
                # box[2] += box[0]
                # box[3] += box[1]

                # [x1, y1, w, h]
                feature = encoder(image, box.reshape(1, -1).copy())

                detections.append(Detection(box, probs[i], feature[0]))

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        bboxes = []
        ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()

            # Convert [x1, y1, x2, y2] to [x, y, w, h]
            # From x2 and y2 to width and height
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            # From top left to center
            bbox[0] += bbox[2] / 2
            bbox[1] += bbox[3] / 2

            bbox[0] /= width
            bbox[1] /= height
            bbox[2] /= width 
            bbox[3] /= height

            bboxes.append(bbox)
            ids.append(track.track_id)

        if len(bboxes) > 0:
            # Draw bounding boxes onto the image
            labels = np.array([2] * len(bboxes))

            image = draw_bounding_boxes(image, bboxes, labels, model.class_names, ids);

        cv2.imshow('2D YOLOv4 SORT', image)

        # Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            clear()
            break

    except KeyboardInterrupt as e:
        clear()
        break
