# APPLIES K CLUSTERING TO MAKE UNIFORM EDGES


import cv2
import numpy as np
from sklearn.cluster import KMeans

N = 400

image = cv2.imread(r"C:\Users\Anusha Rathi\Downloads\bird.png")
# image = cv2.resize(image,(600,600))
image = cv2.resize(image,(80,80))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,100,200)
# edges = cv2.Canny(gray,0,255)

points = np.column_stack(np.where(edges > 0))

kmeans = KMeans(n_clusters=N, random_state=0)
kmeans.fit(points)

drone_positions = kmeans.cluster_centers_

blank = np.zeros((80,80,3), dtype=np.uint8)

for y,x in drone_positions:
    cv2.circle(blank,(int(x),int(y)),3,(0,0,255),-1)

# cv2.imshow("Drone Formation", blank)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(drone_positions)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.target = None
        self.arrived = False


def create_grid_drones(rows, cols, spacing=4):
    drones = []

    for i in range(rows):
        for j in range(cols):
            position = [i * spacing, j * spacing]
            drones.append(Drone(position))

    return drones


def get_targets_from_matrix(matrix, spacing=8):
    targets = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                targets.append(np.array([i * spacing, j * spacing], dtype=float))

    return targets


def assign_targets(drones, targets):

    n_drones = len(drones)
    n_targets = len(targets)

    cost_matrix = np.zeros((n_drones, n_targets))

    for i, drone in enumerate(drones):
        for j, target in enumerate(targets):
            cost_matrix[i][j] = np.linalg.norm(drone.position - target)

    drone_indices, target_indices = linear_sum_assignment(cost_matrix)

    for drone in drones:
        drone.target = None
        drone.arrived = False

    for d, t in zip(drone_indices, target_indices):
        drones[d].target = targets[t]


def compute_separation(drone, drones, safe_distance=2.5):

    force = np.zeros(2)

    for other in drones:

        if other is drone or other.arrived:
            continue

        diff = drone.position - other.position
        distance = np.linalg.norm(diff)

        if 0 < distance < safe_distance:
            force += diff / distance

    return force

def move_toward_target(drone, drones, step_size=0.25):

    if drone.target is None or drone.arrived:
        return

    arrival_radius = 0.6
    slow_radius = 5.0   # within this distance, target force dominates

    target_vec = drone.target - drone.position
    distance = np.linalg.norm(target_vec)

    # snap to target
    if distance < arrival_radius:
        drone.position = drone.target.copy()
        drone.velocity = np.zeros(2)
        drone.arrived = True
        return

    # normalize target direction
    target_dir = target_vec / (distance + 1e-9)

    # get neighbors (only those still flying)
    neighbors = [n for n in get_neighbors(drone, drones) if not n.arrived]

    separation = compute_separation(drone, drones)
    alignment = compute_alignment(drone, neighbors)
    cohesion = compute_cohesion(drone, neighbors)

    # weight swarm forces based on distance to target
    # far from target -> swarm strong
    # near target -> swarm fades out
    swarm_scale = min(1.0, distance / slow_radius)

    direction = (
        1.6 * target_dir +
        swarm_scale * (1.2 * separation + 0.4 * alignment + 0.3 * cohesion)
    )

    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    # smooth slowing near target
    speed = step_size
    if distance < slow_radius:
        speed = step_size * (distance / slow_radius)

    drone.velocity = direction * speed
    drone.position += drone.velocity


def visualize(drones, targets):

    plt.clf()

    active = np.array([d.position for d in drones if not d.arrived])
    finished = np.array([d.position for d in drones if d.arrived])

    target_positions = np.array(targets)

    if len(active) > 0:
        plt.scatter(active[:,0], active[:,1], c="blue", s=30, label="Flying")

    if len(finished) > 0:
        plt.scatter(finished[:,0], finished[:,1], c="green", s=30, label="Arrived")

    # plt.scatter(target_positions[:,0], target_positions[:,1], c="red", marker="x", s=100, label="Targets")

    plt.xlim(-10,80)
    plt.ylim(-10,80)

    plt.legend()
    plt.pause(0.05)

# def generate_target_pattern():

#     targets = []

#     spacing = 6
#     grid_size = 9

#     center = grid_size // 2

#     for i in range(grid_size):
#         for j in range(grid_size):

#             distance = abs(i-center) + abs(j-center)

#             if distance in [2,3,4]:
#                 targets.append([i*spacing, j*spacing])

#     return np.array(targets[:50], dtype=float)

def compute_alignment(drone, neighbors):

    if len(neighbors) == 0:
        return np.zeros(2)

    avg_velocity = np.zeros(2)

    for n in neighbors:
        avg_velocity += n.velocity

    avg_velocity /= len(neighbors)

    alignment_force = avg_velocity - drone.velocity

    return alignment_force

def compute_cohesion(drone, neighbors):

    if len(neighbors) == 0:
        return np.zeros(2)

    center = np.zeros(2)

    for n in neighbors:
        center += n.position

    center /= len(neighbors)

    cohesion_force = center - drone.position

    return cohesion_force

def get_neighbors(drone, drones, radius=8):

    neighbors = []

    for other in drones:

        if other is drone:
            continue

        distance = np.linalg.norm(drone.position - other.position)

        if distance < radius:
            neighbors.append(other)

    return neighbors

if __name__ == "__main__":

    drones = create_grid_drones(20,20)   

    targets = drone_positions

    assign_targets(drones, targets)

    plt.ion()

    steps = 200

    for step in range(steps):

        for drone in drones:
            move_toward_target(drone, drones)

        visualize(drones, targets)

    plt.ioff()
    plt.show()