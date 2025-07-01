# python bbox_heat.py <bbox file>

import pandas as pd
import argparse
import pathlib
import json
import matplotlib.pyplot as plt

def main(bbox_filename = "123_MD_output.json"):
    bbox_file = pathlib.Path(bbox_filename)
    #print(bbox_file)
    # Import file
    with open(bbox_file) as data_file:
        bbox_json = json.load(data_file)
        df = pd.json_normalize(bbox_json, 'images')

    # Get batch name
    batch_id = bbox_filename.split("_")[0]

    # Setup list of all bounding boxes
    all_bboxes = []
    # For each image
    for i, row in df.iterrows():
        #print(i)
        #print(row['detections'])
        row_detections = row['detections']
        print("Number of detections:", len(row_detections))
        # For each detection)
        for j in range(0, len(row_detections)):
            print("Appending Detection", j, row_detections[j])
            all_bboxes.append(row_detections[j])

    all_bboxes_df = pd.json_normalize(all_bboxes)

    # Filter out low confidence
    all_bboxes_df = all_bboxes_df[all_bboxes_df['conf'] >= 0.1]

    # Split into different DFs based on category
    animals_df = all_bboxes_df[all_bboxes_df['category'] == "1"]
    people_df = all_bboxes_df[all_bboxes_df['category'] == "2"]
    vehicles_df = all_bboxes_df[all_bboxes_df['category'] == "3"]

    animals_with_centroids = process_bboxes(animals_df)
    people_with_centroids = process_bboxes(people_df)
    vehicles_with_centroids = process_bboxes(vehicles_df)

    write_plot(animals_with_centroids, people_with_centroids, vehicles_with_centroids)

    # Compound bounding box values to create "heat map"

def process_bboxes(bbox_df):
    # For now, get the centroid on a 1600x1200 image
    x = [max(x[0], x[2]) - min(x[0], x[2]) for x in bbox_df['bbox']]
    y = [max(x[1], x[3]) - min(x[1], x[3]) for x in bbox_df['bbox']]
    centroids = pd.DataFrame({'x': x, 'y': y})
    return centroids

# Write file with batch name
def write_plot(animal_centroids, people_centroids, vehicle_centroids):
    point_alpha = 0.15
    plt.scatter(animal_centroids['x'], animal_centroids['y'], color="blue", alpha= point_alpha)
    plt.scatter(people_centroids['x'], people_centroids['y'], color="yellow", alpha= point_alpha)
    plt.scatter(vehicle_centroids['x'], vehicle_centroids['y'], color="red", alpha= point_alpha)

    # Invert Y axis
    plt.gca().invert_yaxis()
    plt.savefig("bbox_heatmap.jpg")
    plt.close()

# Run Main
if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="bounding box file for batch")
    args = parser.parse_args()

    bbox_file = args.file
    print(bbox_file)
    # If bbox was defined run with that argument
    if bbox_file is not None:
        main(bbox_file)
    else:
        main()
