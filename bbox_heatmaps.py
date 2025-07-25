# python bbox_heatmaps.py <bbox file>
from math import floor

import numpy as np
import pandas as pd
import argparse
import pathlib
import json
import matplotlib.pyplot as plt
from PIL import Image


def main(bbox_filename = "test_md_output.json", min_conf = 0.1):
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
    all_bboxes_df = all_bboxes_df[all_bboxes_df['conf'] >= min_conf]

    # Split into different DFs based on category (reset the indexes to all start at 0)
    animals_df = all_bboxes_df[all_bboxes_df['category'] == "1"].reset_index()
    people_df = all_bboxes_df[all_bboxes_df['category'] == "2"].reset_index()
    vehicles_df = all_bboxes_df[all_bboxes_df['category'] == "3"].reset_index()

    animal_bboxes = process_bboxes(animals_df)
    #people_with_centroids = process_bboxes(people_df)
    #vehicles_with_centroids = process_bboxes(vehicles_df)

    # This can be used when not in production to keep the batch_id intact
    #write_plot(batch_id, animal_bboxes, "animals")

    # When running on HTC, use this:
    write_plot("bbox_heatmap", animal_bboxes, "animals")
    # Compound bounding box values to create "heat map"

def process_bboxes(bbox_df):
    # Create heatmap with zeros and increment by one per pixel.
    heatmap_shape = (600, 800)
    bbox_combined = np.zeros(heatmap_shape)
    #print("BBOX COUNT:", bbox_df['bbox'].count())
    #print(heatmap_shape[0], heatmap_shape[1])
    for i in range(0, bbox_df['bbox'].count()):
        this_bbox = bbox_df['bbox'][i]
        x1pixel,y1pixel,x2pixel,y2pixel = (round(min(this_bbox[0], this_bbox[2]) * (heatmap_shape[1] - 1)),
                                           round(min(this_bbox[1], this_bbox[3]) * (heatmap_shape[0] - 1)),
                                           round(max(this_bbox[0], this_bbox[2]) * (heatmap_shape[1] - 1)),
                                           round(max(this_bbox[1], this_bbox[3]) * (heatmap_shape[0] - 1)))
        #print(i)
        #print(x1pixel,y1pixel,x2pixel,y2pixel)
        # Loop through all pixels covered by this bounding box and add some value to the pixel
        # Could just use 1, but using the bbox confidence might actually give us some additional info?
        for y in range(y1pixel, y2pixel + 1):
            for x in range(x1pixel, x2pixel + 1):
                #print(y, x)
                bbox_combined[y][x] += 1 #bbox_df['conf'][i]

    # Normalize between 0 and 255
    if bbox_df['bbox'].count() > 0:
        # Multiply each item by the scale of bounding boxes processed and normalized by 0-255
        bbox_combined *= 255/bbox_df['bbox'].count()
        #print(bbox_combined.max())
    # print("combined heatmap:", bbox_combined)

    # Plot using PILLOW
    # pil_heatmap = Image.fromarray(bbox_combined)
    # pil_heatmap.show()

    # Plot using matplotlib (vmin/vmax sets color range)
    # Use to see in pycharm
    # plt.show()

    return bbox_combined

# Write file with batch name
def write_plot(file_identifier, bboxes, name, color="hot"):
    # point_alpha = 0.15
    # plt.scatter(animal_centroids['x'], animal_centroids['y'], color="blue", alpha= point_alpha)
    # plt.scatter(people_centroids['x'], people_centroids['y'], color="yellow", alpha= point_alpha)
    # plt.scatter(vehicle_centroids['x'], vehicle_centroids['y'], color="red", alpha= point_alpha)
    # Invert Y axis
    # plt.gca().invert_yaxis()
    # plt.savefig("bbox_heatmap.jpg")
    plt.imshow(bboxes, cmap=color, interpolation='nearest', vmin=0, vmax=255)
    plt.savefig(file_identifier + "_" + name + ".jpg")

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
        main(bbox_file, 0.8)
    else:
        main()
