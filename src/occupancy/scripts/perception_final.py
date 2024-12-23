import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

# Load the point cloud
pcd = o3d.io.read_point_cloud("/Users/yuvan/Desktop/106a_data/screwdriver2/cleaned_scale_screwdriver_cc_format.ply")

# Convert point cloud to numpy arrays
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Step 1: Filter points based on metallic color range
lower_metal = np.array([0.8, 0.6, 0.0])  # Light gray for metallic
upper_metal = np.array([1.0, 0.75, 0.0])  # Darker gray for metallic
mask_metal = np.all((colors >= lower_metal) & (colors <= upper_metal), axis=1)
filtered_metal_points = points[mask_metal]
metal_colors = colors[mask_metal]

# Step 2: Remove outliers from the metallic points
metal_pcd = o3d.geometry.PointCloud()
metal_pcd.points = o3d.utility.Vector3dVector(filtered_metal_points)
metal_pcd.colors = o3d.utility.Vector3dVector(metal_colors)
cleaned_pcd, _ = metal_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01)
cleaned_metal_points = np.asarray(cleaned_pcd.points)

o3d.visualization.draw_geometries([metal_pcd], window_name="Metallic Points", 
                                  width=800, height=600)

# Step 3: Cluster metallic points using DBSCAN and find the largest cluster
db = DBSCAN(eps=0.02, min_samples=10).fit(cleaned_metal_points)
labels = db.labels_
largest_cluster_label = max(set(labels), key=lambda l: np.sum(labels == l))
largest_cluster_points = cleaned_metal_points[labels == largest_cluster_label]

# Step 4: Compute centroid and create bounding box
centroid = np.mean(largest_cluster_points, axis=0)
bbox_min = centroid - np.array([0.095, 0.2, 0.02])  # Half dimensions
bbox_max = centroid + np.array([0.095, 0.2, 0.02])
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)

# Step 5: Filter original point cloud using the bounding box
filtered_pcd = pcd.crop(bbox)

# Step 6: Refine filtered points by clustering again
filtered_points = np.asarray(filtered_pcd.points)
filtered_colors = np.asarray(filtered_pcd.colors)
db_filtered = DBSCAN(eps=0.02, min_samples=10).fit(filtered_points)
labels_filtered = db_filtered.labels_
largest_cluster_label_filtered = max(set(labels_filtered), key=lambda l: np.sum(labels_filtered == l))
final_cluster_points = filtered_points[labels_filtered == largest_cluster_label_filtered]
final_cluster_colors = filtered_colors[labels_filtered == largest_cluster_label_filtered]

# Step 7: Create and save the final cluster point cloud
final_cluster_pcd = o3d.geometry.PointCloud()
final_cluster_pcd.points = o3d.utility.Vector3dVector(final_cluster_points)
final_cluster_pcd.colors = o3d.utility.Vector3dVector(final_cluster_colors)
final_cluster_pcd, _ = final_cluster_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

final_output_path = "test_for_website_dec22.ply"
o3d.io.write_point_cloud(final_output_path, final_cluster_pcd)
print(f"Final filtered point cloud saved to: {final_output_path}")

# Visualize results
o3d.visualization.draw_geometries([final_cluster_pcd])
