# APPLIES K CLUSTERING TO MAKE UNIFORM EDGES


import cv2
import numpy as np
from sklearn.cluster import KMeans

N = 560

image = cv2.imread(r"C:\Users\Gaurav S Shekhawat\Downloads\sample 3.jpg")
image = cv2.resize(image,(600,600))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,100,200)
# edges = cv2.Canny(gray,0,255)

points = np.column_stack(np.where(edges > 0))

kmeans = KMeans(n_clusters=N, random_state=0)
kmeans.fit(points)

drone_positions = kmeans.cluster_centers_

blank = np.zeros((600,600,3), dtype=np.uint8)

for y,x in drone_positions:
    cv2.circle(blank,(int(x),int(y)),3,(0,0,255),-1)

cv2.imshow("Drone Formation", blank)
cv2.waitKey(0)
cv2.destroyAllWindows()