import carla
from sort.sort import Sort

import random
import queue

import cv2
import numpy as np

from what.models.detection.datasets.coco import COCO_CLASS_NAMES

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]]).T

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)

    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img

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

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def get_vanishing_point(p1, p2, p3, p4):

    k1 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])

    vp_x = (k1 * p3[0] - k2 * p1[0] + p1[1] - p3[1]) / (k1 - k2)
    vp_y = k1 * (vp_x - p3[0]) + p3[1]

    return [vp_x, vp_y]

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()

def draw_bounding_boxes(image, boxes, labels, class_names, ids):
    if not hasattr(draw_bounding_boxes, "colours"):
        draw_bounding_boxes.colours = np.random.randint(0, 256, size=(32, 3))

    if len(boxes) > 0:
        assert(boxes.shape[1] == 4)
        boxes = to_numpy(boxes)

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
camera_init_trans = carla.Transform(carla.Location(x=0.5, z=2))

# Create a RGB camera
rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# [Windows Only] Fixes https://github.com/carla-simulator/carla/issues/6085
rgb_camera_bp.set_attribute('image_size_x', '640')
rgb_camera_bp.set_attribute('image_size_y', '640')

camera = world.spawn_actor(rgb_camera_bp, camera_init_trans, attach_to=vehicle)

# Get gamera dimensions and initialise dictionary                       
image_w = rgb_camera_bp.get_attribute("image_size_x").as_int()
image_h = rgb_camera_bp.get_attribute("image_size_y").as_int()

# Callback stores sensor data in a dictionary for use outside callback                         
def camera_callback(image, rgb_image_queue):
    rgb_image_queue.put(np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)))

# Start camera recording
rgb_image_queue = queue.Queue()
camera.listen(lambda image: camera_callback(image, rgb_image_queue))

# Autopilot
vehicle.set_autopilot(True) 

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# Get the attributes from the camera
image_w = rgb_camera_bp.get_attribute("image_size_x").as_int()
image_h = rgb_camera_bp.get_attribute("image_size_y").as_int()
fov = rgb_camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K   = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# Retrieve all the objects of the level
car_objects = world.get_environment_objects(carla.CityObjectLabel.Car) # doesn't have filter by type yet
truck_objects = world.get_environment_objects(carla.CityObjectLabel.Truck) # doesn't have filter by type yet
bus_objects = world.get_environment_objects(carla.CityObjectLabel.Bus) # doesn't have filter by type yet

env_object_ids = []

for obj in (car_objects + truck_objects + bus_objects):
    env_object_ids.append(obj.id)

# Disable all static vehicles
world.enable_environment_objects(env_object_ids, False) 

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
        image =  rgb_image_queue.get()

        # Image preprocessing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        boxes = []
        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:
                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the other vehicle. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA

                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 0:

                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]

                        points_image = []

                        for vert in verts:
                            ray0 = vert - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()

                            if (cam_forward_vec.dot(ray0) > 0):
                                p = get_image_point(vert, K, world_2_camera)
                            else:
                                p = get_image_point(vert, K_b, world_2_camera)

                            points_image.append(p)

                        x_min, x_max = 10000, -10000
                        y_min, y_max = 10000, -10000

                        for edge in edges:
                            p1 = points_image[edge[0]]
                            p2 = points_image[edge[1]]

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            # Both points are out of the canvas
                            if not p1_in_canvas and not p2_in_canvas:
                                continue
                            
                            # Draw 3D Bounding Boxes
                            # cv2.line(image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

                            # Draw 2D Bounding Boxes
                            p1_temp, p2_temp = (p1.copy(), p2.copy())

                            # One of the point is out of the canvas
                            if not (p1_in_canvas and p2_in_canvas):
                                p = [0, 0]

                                # Find the intersection of the edge with the window border
                                p_in_canvas, p_not_in_canvas = (p1, p2) if p1_in_canvas else (p2, p1)
                                k = (p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])

                                x = np.clip(p_not_in_canvas[0], 0, image_w)
                                y = k * (x - p_in_canvas[0]) + p_in_canvas[1]

                                if y >= image_h:
                                    p[0] = (image_h - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = image_h - 1
                                elif y <= 0:
                                    p[0] = (0 - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = 0
                                else:
                                    p[0] = image_w - 1 if x == image_w else 0
                                    p[1] = y

                                p1_temp, p2_temp = (p, p_in_canvas)

                            # Find the rightmost vertex
                            x_max = p1_temp[0] if p1_temp[0] > x_max else x_max
                            x_max = p2_temp[0] if p2_temp[0] > x_max else x_max

                            # Find the leftmost vertex
                            x_min = p1_temp[0] if p1_temp[0] < x_min else x_min
                            x_min = p2_temp[0] if p2_temp[0] < x_min else x_min

                            # Find the highest vertex
                            y_max = p1_temp[1] if p1_temp[1] > y_max else y_max
                            y_max = p2_temp[1] if p2_temp[1] > y_max else y_max

                            # Find the lowest vertex
                            y_min = p1_temp[1] if p1_temp[1] < y_min else y_min
                            y_min = p2_temp[1] if p2_temp[1] < y_min else y_min

                        # Exclude very small bounding boxes
                        if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                            if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                boxes.append(np.array([x_min, y_min, x_max, y_max]))

        # Only draw 2: car, 5: bus, 7: truck
        boxes = np.array(boxes)
        labels = np.array([2] * len(boxes))
        probs = np.array([1.0] * len(boxes))

        if len(boxes) > 0:
            dets = np.concatenate((boxes, probs.reshape((len(probs), -1))), axis=1)

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

                track[0] /= image_w
                track[1] /= image_h
                track[2] /= image_w
                track[3] /= image_h

            # Draw bounding boxes onto the image
            output = draw_bounding_boxes(image, trackers[:, 0 : 4], labels, COCO_CLASS_NAMES, trackers[:, 4])

        cv2.imshow('YOLOv4 Darknet', image)

        # Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            clear()
            break

    except KeyboardInterrupt as e:
        clear()
        break
