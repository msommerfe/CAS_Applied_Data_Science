from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import tensorflow as tf
import os
import pathlib
import shutil


#import ultralytics
print(1)
print(pathlib.Path().resolve())

#copies the results from Ultra lytics. Need to adapt every time you create a new python interpreter
shutil.copytree('/tmp/pycharm_project_498/CAS_Final_Project/runs', '/mnt/c/dev/tmp1')
print(os.getcwd())
print(tf.config.list_physical_devices('GPU'))
workDir = '/mnt/c/dev/Try_EVN_1280x960/'
datasetYamlPath = workDir + 'dataset.yaml'

#model = YOLO('yolov8m.pt')

#results = model.train(data = datasetYamlPath, batch = 32, epochs = 500, imgsz=640)