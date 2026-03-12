# HERE THE SHAPE IS COLOURED SAME AS THE OG IMAGE

import cv2
import numpy as np
from sklearn.cluster import KMeans

N = 200

image = cv2.imread(r"C:\Users\Gaurav S Shekhawat\Downloads\pngtree-color-logo-vector-design-free-png-png-image_6114045.png")
image = cv2.resize(image,(600,600))

mask = np.any(image > [10,10,10], axis=2)

points = np.column_stack(np.where(mask))

kmeans = KMeans(n_clusters=N, random_state=0)
kmeans.fit(points)

drone_positions = kmeans.cluster_centers_

blank = np.zeros((600,600,3), dtype=np.uint8)

for y,x in drone_positions:

    x = int(x)
    y = int(y)

    color = image[y,x]

    cv2.circle(blank,(x,y),3,color.tolist(),-1)

cv2.imshow("Drone Formation", blank)
cv2.waitKey(0)
cv2.destroyAllWindows()
