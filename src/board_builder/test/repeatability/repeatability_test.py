import json
import os
import numpy as np


def compute_mean_corners(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store the sum and count for each marker's corners
    corner_sums = {}
    corner_counts = {}

    # Iterate through each run in the data
    for run in data:
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

    return mean_corners, data


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

            mean_corners, data = compute_mean_corners(input_file)
            mean_distance, std_distance = compute_distances_to_mean(data, mean_corners)

            # Create the output filename by replacing _data.json with _result.json
            output_filename = filename.replace('_data.json', '_results.json')
            output_file = os.path.join(output_dir, output_filename)

            # Write the results to the output file
            write_results_to_file(output_file, mean_corners, mean_distance, std_distance)

            # Print the results for each file
            print(f"Results for {filename}:")
            print(f"Total Mean Distance to Theoretical Values: {mean_distance}")
            print(f"Standard Deviation of Distances: {std_distance}")
            print("\n" + "-"*50 + "\n")