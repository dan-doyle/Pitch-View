"""
K-Means approach
For each pixel in the top half of each player's bounding box, we perform K-means clustering to assign it to one of two clusters. Then, for each player, we look at the labels of all the pixels in the top half of their bounding box, and the player is assigned to the team corresponding to the most common label.
"""

"""
To improve:
- Read paper to see what they did
- Center the bounding box more
- Read up on alternative approaches
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from collections import Counter

def assign_teams(img, boxes, categories, save=False):
    # Load the image
    image = cv2.imread(img)
    copy_image = image.copy()

    # Convert the image to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Player class index
    player_index = 1

    # Collect all pixels in top half of each player's bounding box
    pixels = []
    player_boxes = []

    for i, category in enumerate(categories):
        if category == player_index:
            # Get the top half of the bounding box
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            original_y2 = y2  # Store the original y2 value
            y2 = y1 + (y2 - y1) // 2  # Get the midpoint of y-axis to get top half

            # Get the region of interest
            roi = image[y1:y2, x1:x2]

            # Uniformly scale and center-crop
            roi = resize(roi, (20, 20))
            roi = roi[3:-3, 3:-3]  

            # Average over color channels
            avg_color = np.mean(roi, axis=(0, 1))

            # Add to list
            pixels.append(avg_color)
            player_boxes.append([x1, y1, x2, original_y2])  # Use the original y2 value here

    # Normalize color data
    scaler = StandardScaler()
    normalized_pixels = scaler.fit_transform(pixels)

    dbscan = DBSCAN(eps=1, min_samples=2) 
    dbscan.fit(normalized_pixels)

    team_a_bottom_centers = []
    team_b_bottom_centers = []

    # Apply labels to players
    for i, box in enumerate(player_boxes):
        x1, y1, x2, y2 = box

        label = dbscan.labels_[i]
        bottom_center = ((x1 + x2) // 2, y2)
        
        # Add label to image
        if label == 0:
            team_a_bottom_centers.append(bottom_center)
            cv2.putText(copy_image, 'Team A', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif label == 1:
            team_b_bottom_centers.append(bottom_center)
            cv2.putText(copy_image, 'Team B', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(copy_image, 'Other', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save the image
    if save:
        cv2.imwrite('results/labeled.png', copy_image)

    return team_a_bottom_centers, team_b_bottom_centers

# Update for returning team coords
def assign_teams_k_means(img, boxes, categories, save=False):
    # Load the image
    image = cv2.imread(img)
    copy_image = image.copy()

    # Classes indices should be known, let's assume 'player' has an index of 0
    player_index = 1

    # Collect all average color values in top half of each player's bounding box
    avg_colors = []
    player_boxes = []

    for i, category in enumerate(categories):
        if category == player_index:
            # Get the top half of the bounding box
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y2 = y1 + (y2 - y1) // 2  # Get the midpoint of y-axis to get top half

            # Get the region of interest
            roi = image[y1:y2, x1:x2]
            
            # Resize to 20x20
            roi_resized = cv2.resize(roi, (20, 20))
            
            # Crop the center 16x16
            roi_cropped = roi_resized[2:18, 2:18]
            
            # Get the average color and add to list
            avg_color = np.mean(roi_cropped, axis=(0, 1))
            avg_colors.append(avg_color)
            player_boxes.append([x1, y1, x2, y2])

    # Convert avg_colors to np.array for K-means
    avg_colors = np.array(avg_colors)

    while True:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(avg_colors)
        labels = kmeans.labels_

        # Check if there's a cluster with only one instance
        counter = Counter(labels)
        single_instance_cluster = [label for label, count in counter.items() if count == 1]

        if single_instance_cluster:  # if such a cluster exists
            single_instance_cluster = single_instance_cluster[0]  # get the cluster label
            # exclude the single instance and rerun the clustering
            avg_colors = avg_colors[labels != single_instance_cluster]
            player_boxes = [box for i, box in enumerate(player_boxes) if labels[i] != single_instance_cluster]
        else:
            break  # if no such cluster exists, break the loop

    # Apply labels to players
    for i, box in enumerate(player_boxes):
        x1, y1, x2, y2 = box
        label = kmeans.predict([avg_colors[i]])[0]
        
        # Add label to image
        if label == 0:
            cv2.putText(copy_image, 'Team 1', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif label == 1:
            cv2.putText(copy_image, 'Team 2', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save the image
    if save:
        cv2.imwrite('results/labeled.png', copy_image)