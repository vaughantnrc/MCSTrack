import json
import os
import numpy as np
from src.common.util import register_corresponding_points

FILENAME = 'top_data.json'  # planar_data.json, top_data.json


def transform_point(point, matrix):
    """Applies a 4x4 transformation matrix to a 3D point."""
    point_h = np.array([point[0], point[1], point[2], 1.0])  # convert to homogeneous coordinates
    transformed_point_h = np.matmul(matrix, point_h)
    return transformed_point_h[:3]  # convert back to 3D coordinates

def register_all_runs(data):
    # Use the first run as the reference
    reference_corners = data[0]

    # Initialize a list to store the registered runs
    registered_data = []

    for run in data:
        registered_run = {}
        for marker_id, corners in run.items():
            if marker_id in reference_corners:
                # Get the transformation matrix
                transformation_matrix = register_corresponding_points(reference_corners[marker_id], corners)

                # Apply the transformation to the current marker corners
                registered_corners = []
                for corner in corners:
                    registered_corners.append(transform_point(corner, transformation_matrix).tolist())

                registered_run[marker_id] = registered_corners

        registered_data.append(registered_run)

    return registered_data

def compute_mean_corners(registered_data):
    # Initialize a dictionary to store the sum and count for each marker's corners
    corner_sums = {}
    corner_counts = {}

    # Iterate through each run in the data
    for run in registered_data:
        for marker_id, corners in run.items():
            if marker_id not in corner_sums:
                # Initialize sums and counts for this marker
                corner_sums[marker_id] = np.zeros((4, 3))
                corner_counts[marker_id] = 0

            # Add the current corners to the sum
            corner_sums[marker_id] += np.array(corners)
            corner_counts[marker_id] += 1

    # Compute the mean for each marker's corners
    mean_corners = {}
    for marker_id, sums in corner_sums.items():
        mean_corners[marker_id] = (sums / corner_counts[marker_id]).tolist()

    return mean_corners

def compute_distances_to_mean(data, mean_corners):
    distances = []

    for run in data:
        for marker_id, corners in run.items():
            mean_corners_marker = np.array(mean_corners[marker_id])
            for i, corner in enumerate(corners):
                distance = np.linalg.norm(np.array(corner) - mean_corners_marker[i])
                distances.append(distance)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return mean_distance, std_distance

def write_results_to_file(output_file, mean_corners, mean_distance, std_distance):
    results = {
        "mean_corners": mean_corners,
        "mean_distance_to_theoretical_values": mean_distance,
        "standard_deviation_of_distances": std_distance
    }

    # Write the results to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'collected_data')
    output_dir = os.path.join(script_dir, 'repeatability_test_results')

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each JSON file in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(results_dir, filename)

            with open(input_file, 'r') as file:
                data = json.load(file)

            registered_data = register_all_runs(data)
            mean_corners = compute_mean_corners(registered_data)
            mean_distance, std_distance = compute_distances_to_mean(registered_data, mean_corners)

            output_filename = filename.replace('_data.json', '_results.json')
            output_file = os.path.join(output_dir, output_filename)

            write_results_to_file(output_file, mean_corners, mean_distance, std_distance)

            # Print the results for each file
            print(f"Results for {filename}:")
            print(f"Total Mean Distance to Theoretical Values: {mean_distance}")
            print(f"Standard Deviation of Distances: {std_distance}")
            print("\n" + "-"*50 + "\n")
