import cv2
import numpy as np
import matplotlib.pyplot as plt


def sample_contour_uniform(contour, spacing_pixels):
    """
    Sample points along a contour with approximately equal spacing.
    """
    contour = contour.squeeze()

    # compute distances between consecutive contour points
    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))

    cumulative = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative[-1]

    n_points = max(1, int(total_length / spacing_pixels))

    target_distances = np.linspace(0, total_length, n_points)

    sampled = []

    for d in target_distances:
        idx = np.searchsorted(cumulative, d)
        idx = min(idx, len(contour) - 1)
        sampled.append(contour[idx])

    return np.array(sampled)


def generate_drone_formation(image_path, d_min_meters, pixel_to_meter_scale):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found")

    # smooth to remove noise
    # blurred = cv2.GaussianBlur(img, (5,5), 0)

    edges = cv2.Canny(img, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    drone_coords = []

    # convert spacing from meters to pixels
    spacing_pixels = d_min_meters / pixel_to_meter_scale

    for contour in contours:

        if len(contour) < 20:
            continue

        sampled_points = sample_contour_uniform(contour, spacing_pixels)

        for pt in sampled_points:

            x_pixel = pt[0]
            y_pixel = pt[1]

            real_x = x_pixel * pixel_to_meter_scale
            real_y = 0
            real_z = (img.shape[0] - y_pixel) * pixel_to_meter_scale

            drone_coords.append((real_x, real_y, real_z))

    drone_coords = np.array(drone_coords)

    return len(drone_coords), drone_coords


# ------------------------------
# Execution
# ------------------------------

IMAGE_PATH = r"C:\Users\Gaurav S Shekhawat\Downloads\sample 3.jpg"

MIN_DISTANCE = 2.5
SCALE_FACTOR = 0.1

n_required, coordinates = generate_drone_formation(
    IMAGE_PATH,
    MIN_DISTANCE,
    SCALE_FACTOR
)

# print("Number of drones:", n_required)
# print("First coordinates:\n", coordinates[:5])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    coordinates[:,0],
    coordinates[:,1],
    coordinates[:,2],
    c='blue',
    s=10
)

ax.set_xlabel("X (meters)")
ax.set_ylabel("Depth Y")
ax.set_zlabel("Altitude Z")
ax.set_title(f"{n_required} Drone Formation")

plt.show()