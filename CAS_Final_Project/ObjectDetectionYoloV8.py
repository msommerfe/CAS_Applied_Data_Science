from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import tensorflow as tf

#import ultralytics

print(tf.config.list_physical_devices('GPU'))
workDir = '/mnt/c/dev/Try_EVN_1280x960/'
datasetYamlPath = workDir + 'dataset.yaml'

model = YOLO('yolov8m.pt')

results = model.train(data = datasetYamlPath, batch = 32, epochs = 500, imgsz=640)