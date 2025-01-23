import enum
import numpy as np
import json
from rrt_algorithms.rrt.rrt_base import RRTBase
from rrt_algorithms.utilities.geometry import steer
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
import os


class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4


class RRTConnect(RRTBase):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)
        self.swapped = False

    def swap_trees(self):
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.swapped = not self.swapped

    def unswap(self):
        if self.swapped:
            self.swap_trees()

    def extend(self, tree, x_rand, enforce_waypoint=False, waypoint=None, restrict_side=None):
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = steer(x_nearest, x_rand, self.q)

        # Enforce waypoint and side constraints
        if enforce_waypoint and waypoint:
            if np.linalg.norm(np.array(x_new) - np.array(waypoint)) < self.q:
                x_new = waypoint
                if restrict_side is not None:
                    if restrict_side == "left" and x_new[0] > waypoint[0]:
                        return x_new, Status.TRAPPED
                    elif restrict_side == "right" and x_new[0] < waypoint[0]:
                        return x_new, Status.TRAPPED

        if self.connect_to_point(tree, x_nearest, x_new):
            if np.abs(np.sum(np.array(x_new) - np.array(x_rand))) < 1e-2:
                return x_new, Status.REACHED
            return x_new, Status.ADVANCED
        return x_new, Status.TRAPPED

    def connect(self, tree, x):
        S = Status.ADVANCED
        while S == Status.ADVANCED:
            x_new, S = self.extend(tree, x)
        return x_new, S

    def rrt_connect(self, waypoint=None, restrict_side=None):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, self.x_goal, None)

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            enforce_waypoint = False

            if waypoint and np.linalg.norm(np.array(x_rand) - np.array(waypoint)) < self.q:
                enforce_waypoint = True

            x_new, status = self.extend(0, x_rand, enforce_waypoint, waypoint, restrict_side)
            if status != Status.TRAPPED:
                x_new, connect_status = self.connect(1, x_new)
                if connect_status == Status.REACHED:
                    self.unswap()
                    first_part = self.reconstruct_path(0, self.x_init, self.get_nearest(0, x_new))
                    second_part = self.reconstruct_path(1, self.x_goal, self.get_nearest(1, x_new))
                    second_part.reverse()
                    return first_part + second_part
            self.swap_trees()
            self.samples_taken += 1


repo_dir = os.path.dirname(os.path.abspath(__file__))

detections_file_path = os.path.join(repo_dir, "detections.json")

with open(detections_file_path, "r") as file:
    data = json.load(file)

detections = data[0]["detections"]
obstacles = [
    (bbox[0], bbox[1], bbox[2], bbox[3])
    for detection in detections
    if detection["class"] == "Obstacles" and detection["confidence"] > 0.5
    for bbox in detection["bbox"]
]
cones = [
    (bbox[0], bbox[1], bbox[2], bbox[3])
    for detection in detections
    if detection["class"] == "Cones" and detection["confidence"] > 0.5
    for bbox in detection["bbox"]
]

X_dimensions = np.array([(0, 1280), (0, 1280)])

all_obstacles = np.array(obstacles + cones)

x_init = (1280, 0)
x_goal = (0, 1280)

cone1 = cones[0]
cone2 = cones[1]

cone1_center = ((cone1[0] + cone1[2]) / 2, (cone1[1] + cone1[3]) / 2)
cone2_center = ((cone2[0] + cone2[2]) / 2, (cone2[1] + cone2[3]) / 2)

midpoint = ((cone1_center[0] + cone2_center[0]) / 2, (cone1_center[1] + cone2_center[1]) / 2)

waypoint = midpoint

direction_vector = np.array([cone2_center[0] - cone1_center[0], cone2_center[1] - cone1_center[1]])
direction_vector = direction_vector / np.linalg.norm(direction_vector)
perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
offset_distance = 50

left_waypoint = (
    waypoint[0] + perpendicular_vector[0] * offset_distance,
    waypoint[1] + perpendicular_vector[1] * offset_distance,
)
right_waypoint = (
    waypoint[0] - perpendicular_vector[0] * offset_distance,
    waypoint[1] - perpendicular_vector[1] * offset_distance,
)

q = 50
r = 5
max_samples = 2048
prc = 0.1

X = SearchSpace(X_dimensions, all_obstacles)

goal_vector = np.array([x_goal[0] - midpoint[0], x_goal[1] - midpoint[1]])

cross_product = direction_vector[0] * goal_vector[1] - direction_vector[1] * goal_vector[0]

if cross_product > 0:
    rrt_connect = RRTConnect(X, q, x_init, right_waypoint, max_samples, r, prc)
    path_to_right_waypoint = rrt_connect.rrt_connect(waypoint=right_waypoint, restrict_side="right")

    rrt_connect = RRTConnect(X, q, right_waypoint, waypoint, max_samples, r, prc)
    path_to_midpoint = rrt_connect.rrt_connect(waypoint=waypoint)

    rrt_connect = RRTConnect(X, q, waypoint, left_waypoint, max_samples, r, prc)
    path_to_left_waypoint = rrt_connect.rrt_connect(waypoint=left_waypoint, restrict_side="left")

    rrt_connect = RRTConnect(X, q, left_waypoint, x_goal, max_samples, r, prc)
    path_to_goal = rrt_connect.rrt_connect()

    final_path = path_to_right_waypoint + path_to_midpoint + path_to_left_waypoint + path_to_goal

elif cross_product < 0:
    rrt_connect = RRTConnect(X, q, x_init, left_waypoint, max_samples, r, prc)
    path_to_left_waypoint = rrt_connect.rrt_connect(waypoint=left_waypoint, restrict_side="left")

    rrt_connect = RRTConnect(X, q, left_waypoint, waypoint, max_samples, r, prc)
    path_to_midpoint = rrt_connect.rrt_connect(waypoint=waypoint)

    rrt_connect = RRTConnect(X, q, waypoint, right_waypoint, max_samples, r, prc)
    path_to_right_waypoint = rrt_connect.rrt_connect(waypoint=right_waypoint, restrict_side="right")

    rrt_connect = RRTConnect(X, q, right_waypoint, x_goal, max_samples, r, prc)
    path_to_goal = rrt_connect.rrt_connect()

    final_path = path_to_left_waypoint + path_to_midpoint + path_to_right_waypoint + path_to_goal
    
plot = Plot("Dynamic Drone Path Planning with RRTConnect and Waypoints")
plot.plot_tree(X, rrt_connect.trees)
if final_path is not None:
    plot.plot_path(X, final_path)
plot.plot_obstacles(X, all_obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.plot_waypoint(X, waypoint)
plot.plot_waypoint(X, left_waypoint)
plot.plot_waypoint(X, right_waypoint)
plot.draw(auto_open=True)
