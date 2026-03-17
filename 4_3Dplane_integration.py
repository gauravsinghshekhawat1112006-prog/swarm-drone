import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D

N = 400
Z_TARGET = 20   

image = cv2.imread(r"C:\Users\Anusha Rathi\Downloads\phoenix.png")
image = cv2.resize(image, (80, 80))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# convert (row, col) → (x, y)
points = np.column_stack(np.where(edges > 0))[:, ::-1]

kmeans = KMeans(n_clusters=N, random_state=0)
kmeans.fit(points)

drone_positions_2d = kmeans.cluster_centers_

drone_positions = np.column_stack(
    (drone_positions_2d, np.full(len(drone_positions_2d), Z_TARGET))
)


class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.target = None
        self.arrived = False


def create_grid_drones(rows, cols, spacing=4):
    drones = []
    for i in range(rows):
        for j in range(cols):
            drones.append(Drone([i * spacing, j * spacing, 0]))  
    return drones

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

def move_toward_target(drone, step_size=0.4):
    if drone.target is None or drone.arrived:
        return

    target_vec = drone.target - drone.position
    distance = np.linalg.norm(target_vec)

    if distance < 0.5:
        drone.position = drone.target.copy()
        drone.velocity = np.zeros(3)
        drone.arrived = True
        return

    direction = target_vec / (distance + 1e-9)

    drone.velocity = direction * step_size
    drone.position += drone.velocity

def visualize_3d(ax, drones):
    ax.clear()

    active = np.array([d.position for d in drones if not d.arrived])
    finished = np.array([d.position for d in drones if d.arrived])

    if len(active) > 0:
        ax.scatter(active[:, 0], active[:, 1], active[:, 2],
                   c='blue', s=20, label='Flying')

    if len(finished) > 0:
        ax.scatter(finished[:, 0], finished[:, 1], finished[:, 2],
                   c='green', s=20, label='Arrived')

    # set limits dynamically
    all_points = np.array([d.position for d in drones])

    ax.set_xlim(np.min(all_points[:, 0]) - 5, np.max(all_points[:, 0]) + 5)
    ax.set_ylim(np.min(all_points[:, 1]) - 5, np.max(all_points[:, 1]) + 5)
    ax.set_zlim(0, Z_TARGET + 5)

    # make XY plane horizontal
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=30, azim=45)  # nice angle

    plt.pause(0.05)


if __name__ == "__main__":

    drones = create_grid_drones(20, 20)
    targets = drone_positions

    assign_targets(drones, targets)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    steps = 200

    for step in range(steps):
        for drone in drones:
            move_toward_target(drone)

        visualize_3d(ax, drones)

    plt.ioff()
    plt.show()