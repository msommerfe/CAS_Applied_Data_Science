from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import tensorflow as tf
import os
import pathlib
from time import sleep
import shutil
import numpy as np
import csv
import matplotlib.pyplot as plt


print(os.getcwd())
print(tf.config.list_physical_devices('GPU'))



workDir = '/mnt/c/dev/Try_EVN_1280x960/'
datasetYamlPath = workDir + 'dataset.yaml'

directory_with_val_images = os.path.join(workDir, "validate/images")
#directory_with_val_images = os.path.join(workDir, "all_pics")
#pathBestWeights = os.path.join(workDir, "runs/longRun_240Epochs/detect/train/weights/best.pt")
pathBestWeights = '/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/Weights/YoloV8/best.pt'


def train_yolo_model():
    model = YOLO('yolov8m.pt')
    results = model.train(data=datasetYamlPath, batch=32, epochs=2500, imgsz=928, patience=500)
    # copies the results from Ultra lytics. Need to adapt every time you create a new python interpreter
    shutil.copytree('/tmp/pycharm_project_94/CAS_Final_Project/runs', '/mnt/c/dev/tmp250724')
    return model

def load_yolo_model(pathBestWeights):
    model = YOLO(pathBestWeights)
    return model


def detect_images_in_directory(dir, yolo_model, doPlot = True):
    # Detects objects in all images in dirPath.
    yoloRresults = yolo_model(dir, conf = 0.7)

    if doPlot:
        # Display all images
        for r in yoloRresults:
            plt.imshow(np.squeeze(r.plot()))
            plt.show()
            sleep(2)

    return yoloRresults


def create_cropped_images(yoloRresults, doPlot = True):
    ultralytics_crop_objects = []
    # Showing the croped pictures
    for r in yoloRresults:
        img = r.orig_img

        # Extract bounding boxes
        boxes = r.boxes.xyxy.tolist()

        # Iterate through the bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            ultralytics_crop_objects.append(img[int(y1):int(y2), int(x1):int(x2)])

    if doPlot:
        # Plot croped image
        for im in ultralytics_crop_objects:
            plt.imshow(im)
            plt.show()
            sleep(1)
    return ultralytics_crop_objects


def save_cropped_images(dir, ultralytics_crop_objects):
    i = 0
    filenames = []
    for cropped_img in ultralytics_crop_objects:
        filename = os.path.join(dir, str(i) + '.png')
        filenames.append(str(i) + '.png')
        plt.imsave(filename, cropped_img)
        i = i+1
    print(type(filenames))

    with open(os.path.join(dir,'annotations.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter= ' ')
        for line in filenames:
            writer.writerow([str(line), ''])

    return 0




#model = train_yolo_model()
model = load_yolo_model(pathBestWeights)
detection_results = detect_images_in_directory(directory_with_val_images, model, doPlot = True)
#cropped_images = create_cropped_images(detection_results, doPlot = False)
#save_cropped_images(os.path.join(workDir, "cropped_images"), cropped_images)
