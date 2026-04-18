import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# -------------------------
# PARAMETERS
# -------------------------
N = 300
Z_TARGET = 20
IMAGE_SIZE = 600
GRID_SIZE = 20

# -------------------------
# LOAD IMAGE
# -------------------------
image = cv2.imread(r"C:\Users\Gaurav S Shekhawat\Downloads\sample 2.jpg")
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 100, 200)

# -------------------------
# EXTRACT EDGE POINTS
# -------------------------
points = np.column_stack(np.where(edges > 0))[:, ::-1]

# N = min(N, len(points))

kmeans = KMeans(n_clusters=N, random_state=0)
kmeans.fit(points)

centers = kmeans.cluster_centers_

targets = np.column_stack(
    (centers, np.full(len(centers), Z_TARGET))
)

# -------------------------
# DRONE CLASS
# -------------------------
class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.target = None
        self.arrived = False

# -------------------------
# CREATE GRID
# -------------------------
spacing = IMAGE_SIZE / GRID_SIZE

def create_grid_drones(rows, cols):
    drones = []
    for i in range(rows):
        for j in range(cols):
            drones.append(Drone([i * spacing, j * spacing, 0]))
    return drones

# -------------------------
# ASSIGN TARGETS (Hungarian)
# -------------------------
def assign_targets(drones, targets):

    n = min(len(drones), len(targets))
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost_matrix[i][j] = np.linalg.norm(
                drones[i].position - targets[j]
            )

    drone_idx, target_idx = linear_sum_assignment(cost_matrix)

    for d, t in zip(drone_idx, target_idx):
        drones[d].target = targets[t]

# -------------------------
# SMOOTH MOVEMENT
# -------------------------
def move_toward_target(drone):

    if drone.target is None or drone.arrived:
        return

    target_vec = drone.target - drone.position
    distance = np.linalg.norm(target_vec)

    # relaxed arrival condition
    if distance < 2.0:
        drone.position = drone.target.copy()
        drone.velocity = np.zeros(3)
        drone.arrived = True
        return

    direction = target_vec / (distance + 1e-9)

    # smooth slowing near target
    slow_radius = 50
    max_speed = 2

    if distance < slow_radius:
        speed = max_speed * (distance / slow_radius)
    else:
        speed = max_speed

    # smooth velocity update (prevents jerks)
    desired_velocity = direction * speed

    # smoothing factor (VERY IMPORTANT)
    alpha = 0.2
    drone.velocity = (1 - alpha) * drone.velocity + alpha * desired_velocity

    drone.position += drone.velocity

# -------------------------
# VISUALIZATION
# -------------------------
def visualize_3d(ax, drones):

    ax.clear()

    active = np.array([d.position for d in drones if not d.arrived])
    finished = np.array([d.position for d in drones if d.arrived])

    if len(active) > 0:
        ax.scatter(active[:,0], active[:,1], active[:,2],
                   c='blue', s=5)

    if len(finished) > 0:
        ax.scatter(finished[:,0], finished[:,1], finished[:,2],
                   c='green', s=5)

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(0, IMAGE_SIZE)
    ax.set_zlim(0, Z_TARGET + 10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=-100, azim=-90)

    plt.pause(0.03)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    drones = create_grid_drones(GRID_SIZE, GRID_SIZE)

    assign_targets(drones, targets)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    steps = 400   # more steps for convergence

    for _ in range(steps):
        for drone in drones:
            move_toward_target(drone)

        visualize_3d(ax, drones)

    plt.ioff()
    plt.show()