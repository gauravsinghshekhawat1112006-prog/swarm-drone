import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D

# ---------------- PARAMETERS ----------------
N = 400
Z_TARGET = 20
IMAGE_SIZE = 300
import os

image_paths = []

for i in range(1, 10):
    filename = f"E:\\mars\\test images\\try2\\frame_{i:02d}.png"
    # filename = f"E:\mars\test images\frame_01.png"
    # filename = f"E:\mars\test images\try2\frame_01.png"

    if os.path.exists(filename):
        image_paths.append(filename)


# image_paths = [
#     r"C:\Users\Anusha Rathi\Downloads\phoenix.png",
#     r"C:\Users\Anusha Rathi\Downloads\wlw.png",
#     r"C:\Users\Anusha Rathi\Downloads\bird.png"
# ]

# ---------------- PROCESS IMAGE ----------------
def get_targets_from_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    points = np.column_stack(np.where(edges > 0))[:, ::-1]

    n = min(N, len(points))

    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(points)

    centers = kmeans.cluster_centers_

    # Add Z dimension
    return np.column_stack((centers, np.full(len(centers), Z_TARGET)))

# ---------------- DRONE CLASS ----------------
class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.target = None
        self.arrived = False

# ---------------- CREATE GRID ----------------
GRID_SIZE = 20
space = IMAGE_SIZE / GRID_SIZE

def create_grid_drones(rows, cols):
    drones = []
    for i in range(rows):
        for j in range(cols):
            drones.append(Drone([i * space, j * space, 0]))  # start at z=0
    return drones

# ---------------- ASSIGN TARGETS ----------------
def assign_targets(drones, targets):
    cost_matrix = np.zeros((len(drones), len(targets)))

    for i, drone in enumerate(drones):
        for j, target in enumerate(targets):
            cost_matrix[i][j] = np.linalg.norm(drone.position - target)

    drone_idx, target_idx = linear_sum_assignment(cost_matrix)

    for d, t in zip(drone_idx, target_idx):
        drones[d].target = targets[t]
        drones[d].arrived = False

# ---------------- MOVEMENT ----------------
def move_drones(drones, step_size=1):
    all_arrived = True

    for drone in drones:
        if drone.target is None:
            continue

        target_vec = drone.target - drone.position
        dist = np.linalg.norm(target_vec)

        if dist < 1:
            drone.position = drone.target.copy()
            drone.arrived = True
        else:
            direction = target_vec / (dist + 1e-9)
            drone.position += direction * step_size
            all_arrived = False

    return all_arrived

# ---------------- 3D VISUALIZATION ----------------
def visualize_3d(ax, drones, title=""):
    ax.clear()

    active = np.array([d.position for d in drones if not d.arrived])
    finished = np.array([d.position for d in drones if d.arrived])

    # flying drones
    if len(active) > 0:
        ax.scatter(active[:, 0], active[:, 1], active[:, 2],
                   c='blue', s=10)

    # arrived drones
    if len(finished) > 0:
        ax.scatter(finished[:, 0], finished[:, 1], finished[:, 2],
                   c='green', s=10)

    # axis limits
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(0, IMAGE_SIZE)
    ax.set_zlim(0, Z_TARGET + 10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    # fixed viewing angle
    ax.view_init(elev=-100, azim=-90)

    plt.pause(0.01)

# ---------------- MAIN ----------------
if __name__ == "__main__":

    drones = create_grid_drones(20, 20)

    # Precompute all image targets
    all_targets = [get_targets_from_image(p) for p in image_paths]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, targets in enumerate(all_targets):

        print(f"Transitioning to Image {idx+1}")

        assign_targets(drones, targets)

        # Move drones to next formation
        for step in range(150):
            done = move_drones(drones)
            visualize_3d(ax, drones, f"Image {idx+1} | Step {step}")

            if done:
                break

        # Hold formation
        for _ in range(40):
            visualize_3d(ax, drones, f"Stable Image {idx+1}")

    plt.ioff()
    plt.show()