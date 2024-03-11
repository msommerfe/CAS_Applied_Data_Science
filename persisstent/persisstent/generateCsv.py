from os import listdir
from os.path import isfile, join

path = "Try_EVN_1280x960/"
ml_use = "train"
images = [f for f in listdir(path+ml_use+"/images") if isfile(join(path+ml_use+"/images", f))]
print(images)
with (open("train.csv","a") as f):
    for image in images:
        print(image)
        imagePath = path+ml_use+"/images/"+image
        print(imagePath)
        labelPath = path+ml_use+"/labels/"+image.replace(".jpg",".txt")
        print(labelPath)
        if isfile(labelPath):
            with open(labelPath,"r") as label:
                readline = label.readline().replace("\n","")
                spliten = readline.split(" ")
                l = ""
                if spliten[0] == "15":
                    l = "EVN_H"
                if spliten[0] == "16":
                    l = "EVN_V"
                if spliten[0] == "17":
                    l = "EVN_HV"
                if l != "":
                    lineToWrite = "TRAINING, gs://persisstent/Try_EVN_1280x960/" + ml_use + "/images/" + image + "," + l + ","
                    lineToWrite = lineToWrite + spliten[1] + ","+ spliten[2] + ",,,"+ spliten[3] + ","+ spliten[4] + ",,\n"
                    f.write(lineToWrite)
                    # 16 0.424583 0.425373 0.054167 0.067164