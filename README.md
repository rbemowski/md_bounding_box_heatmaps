# MegaDetector bounding box heatmap

Given an output from megadetector, this script will build and output a 
heatmap of he bounding boxes found in the .json file. This can be very
helpful when performing analysis of a camera trap site.

## How to run

The main goal of this project is to run on High Throughput Computing (HTC) 
resources. Eventually I plan to make it just as easy to use on a local 
machine, but that is not the focus now. 

## File descriptions

- bbox_heatmaps.py - Logic for creating the heatmaps
- bbox_heatmaps.sub - Used for HTCondor High Throughput Compute (HTC) jobs
- bbox_heatmaps.sh - Used to kick off jobx once on the HTC execution points (EP)
