import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# -------------------------
# PARAMETERS
# -------------------------
N = 2000
Z_TARGET = 20
IMAGE_SIZE = 300
# GRID_SIZE = 20

# -------------------------
# LOAD IMAGE
# -------------------------
image = cv2.imread(r"C:\Users\Gaurav S Shekhawat\Downloads\virat kohli.webp")
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

# -------------------------
# CREATE MASK (OBJECT ONLY)
# -------------------------
mask = np.any(image > [10,10,10], axis=2)

points = np.column_stack(np.where(mask))   # (y,x)

# N = min(N, len(points))

# -------------------------
# KMEANS FOR DRONE POSITIONS
# -------------------------
kmeans = KMeans(n_clusters=N, random_state=0)
kmeans.fit(points)

centers = kmeans.cluster_centers_

# -------------------------
# CREATE TARGETS WITH COLOR
# -------------------------
targets = []

for y, x in centers:

    xi = int(x)
    yi = int(y)

    color = image[yi, xi]  # BGR

    targets.append({
        "pos": np.array([x, y, Z_TARGET]),
        "color": color
    })

# -------------------------
# DRONE CLASS
# -------------------------
class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.target = None
        self.arrived = False
        self.color = np.array([255,0,0])   # default blue
        self.target_color = None

# -------------------------
# CREATE GRID
# -------------------------
# spacing = IMAGE_SIZE / GRID_SIZE

def create_grid_drones(N, IMAGE_SIZE):

    rows = int(np.sqrt(N))
    cols = int(np.ceil(N / rows))

    spacing_x = IMAGE_SIZE / cols
    spacing_y = IMAGE_SIZE / rows

    drones = []

    for i in range(rows):
        for j in range(cols):
            x = j * spacing_x
            y = i * spacing_y

            drones.append(Drone([x, y, 0]))

    return drones[:N]

# -------------------------
# ASSIGN TARGETS
# -------------------------
def assign_targets(drones, targets):

    n = min(len(drones), len(targets))
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost_matrix[i][j] = np.linalg.norm(
                drones[i].position - targets[j]["pos"]
            )

    drone_idx, target_idx = linear_sum_assignment(cost_matrix)

    for d, t in zip(drone_idx, target_idx):
        drones[d].target = targets[t]["pos"]
        drones[d].target_color = targets[t]["color"]

# -------------------------
# SMOOTH MOVEMENT
# -------------------------
def move_toward_target(drone):

    if drone.target is None or drone.arrived:
        return

    target_vec = drone.target - drone.position
    distance = np.linalg.norm(target_vec)

    # arrival condition
    if distance < 2.0:
        drone.position = drone.target.copy()
        drone.velocity = np.zeros(3)
        drone.arrived = True

        # 🔥 apply final color
        if drone.target_color is not None:
            drone.color = drone.target_color

        return

    direction = target_vec / (distance + 1e-9)

    slow_radius = 50
    max_speed = 1

    if distance < slow_radius:
        speed = max_speed * (distance / slow_radius)
    else:
        speed = max_speed

    desired_velocity = direction * speed

    alpha = 0.2  # smoothing
    drone.velocity = (1 - alpha) * drone.velocity + alpha * desired_velocity

    drone.position += drone.velocity

# -------------------------
# VISUALIZATION
# -------------------------
def visualize_3d(ax, drones):

    ax.clear()

    active = [d for d in drones if not d.arrived]
    finished = [d for d in drones if d.arrived]

    if len(active) > 0:
        pos = np.array([d.position for d in active])
        ax.scatter(pos[:,0], pos[:,1], pos[:,2],
                   c='blue', s=5)

    if len(finished) > 0:
        pos = np.array([d.position for d in finished])

        # convert BGR → RGB
        colors = [d.color[::-1] / 255 for d in finished]

        ax.scatter(pos[:,0], pos[:,1], pos[:,2],
                   c=colors, s=10)

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

    # drones = create_grid_drones(GRID_SIZE, GRID_SIZE)
    drones = create_grid_drones(N, IMAGE_SIZE)

    assign_targets(drones, targets)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    steps = 400

    for _ in range(steps):
        for drone in drones:
            move_toward_target(drone)

        visualize_3d(ax, drones)

    plt.ioff()
    plt.show()