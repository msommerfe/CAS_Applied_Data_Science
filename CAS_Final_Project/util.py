# Add the total path to image name
import string
import tensorflow as tf
import json
import numpy as np
import csv

def get_global_var():
    MAX_HIGHT = 64
    MAX_WIDTH = 128

    #I found plenty of diferent label file formats.These 3 Formats I tested quite a lot
    IMG_FOLDER = 'G:/My Drive/development/datasets/OCR/MNIST_words_cropped/images/'
    LABELS_File = 'G:/My Drive/development/datasets/OCR/MNIST_words_cropped/annotations.json'
    #IMG_FOLDER = '/content/tr_synth_100K_cropped/images/'
    #LABELS_File = '/content/tr_synth_100K_cropped/annotations.txt'
    #IMG_FOLDER = '/content/tr_synth_100K_cropped/images/'
    #LABELS_File = '/content/SVHN_annotations.out'

    #Dont use # in the alphabet. if you need you need to change the fillup char
    #ALPHABETS = '#'+ string.digits + string.ascii_letters + '!?.-()+ '
    ALPHABETS = '#'+ string.digits + string.ascii_uppercase + '- '

    MAX_STR_LEN = 24 # max length of input labels
    NUM_OF_CHARACTERS = len(ALPHABETS) + 1 # +1 for ctc pseudo blank
    NUM_OF_TIMESTAMPS = 24 # max length of predicted labels
    BATCH_SIZE = 32
    return MAX_HIGHT, MAX_WIDTH, IMG_FOLDER, LABELS_File, ALPHABETS, MAX_STR_LEN, NUM_OF_CHARACTERS, NUM_OF_TIMESTAMPS, BATCH_SIZE

MAX_HIGHT, MAX_WIDTH, IMG_FOLDER, LABELS_File, ALPHABETS, MAX_STR_LEN, NUM_OF_CHARACTERS, NUM_OF_TIMESTAMPS, BATCH_SIZE = get_global_var()

def make_total_path(imgName):
    return IMG_FOLDER + imgName


### Converting Chars to nums is better for ML
def label_to_num(label, padding = True):
    label_num = []
    for ch in label:
        #This line assignes all characters that are not in the Alphabet a -1. It means there is no character. An alternative could be assignt it to a pseudo character that represents all characters, that are not in the alphabet
        #I use a pseudo character that represents all characters, that are not in the alphabet
        label_num.append(ALPHABETS.find(ch) if ALPHABETS.find(ch)!=-1 else ALPHABETS.find('#'))

    if padding == True:
        return_label = tf.keras.utils.pad_sequences([label_num], maxlen= MAX_STR_LEN, value= -1, padding = "post")
        return return_label[0]
    else:
        return label_num

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=ALPHABETS[ch]
    return ret
#print(label_to_num("aA01"))
#num_to_label(label_to_num("aA01"))


#Reading the key values out (x und y) of a json {Image_filename, "test on Image"}
def import_json_label_file(path = LABELS_File):
    with open(path) as f:
        data = list(json.load(f).items())
    return np.array(data)

#keyVal = import_json_label_file()[:8000]
#print(keyVal)

#Reading the key values out (x und y) of a csv / txt ("Image_filename" "text on Image")
def import_txt_csv_label_file(path = LABELS_File):
    with open(path, "r") as f:
        data = list(csv.reader(f, delimiter=" "))
    return np.array(data)

    #Reduced dataset, change it to all for real training
#keyVal = import_txt_csv_label_file()[:8000]
#print(keyVal)



#Reading the key values out (x und y) of a spezific *.out file ("Image_filename that includes the labels" "text on Image")
def import_txt_csv_label_file(path = LABELS_File):
    with open(LABELS_File, "r") as f:
        data = list(csv.reader(f, delimiter=" "))

    data1 = list(map(lambda element: str(element[0]).split("."), np.array(data)))
    data2 = list(map(lambda element: (element[0] +"." + element[1],element[2]), data1))
    return np.unique(data2, axis=0)

    #Reduced dataset, change it to all for real training
#keyVal = import_txt_csv_label_file()[:8000]
#print(keyVal.shape)
#print(keyVal)


#make total path instead of just image name
def make_total_path_for_all_image_names(keyVal):
    new_key_val = []

    path = np.array([make_total_path(imgName) for imgName in keyVal[:,0]])
    for index in range(len(path)):
        new_key_val.append([path[index], keyVal[index,1]])

    return np.array(new_key_val)

#Delete all values that are not in the alphabet
def delete_key_values_that_not_in_alphabet(key_val):
    i = 0
    clean_matrix = np.full((len(key_val)), True)
    for item in key_val:
        #print(item[1])
        for char in item[1]:
            clean_matrix[i] = True
            if ALPHABETS.find(char)==-1:
                clean_matrix[i] = False
                break
        i = i+1

    return key_val[clean_matrix]

#Delete all Images with a tooo small aspect ratio, because the throughing an error
def delete_key_values_with_too_small_aspect_ratio(key_val):
    clean_matrix = np.full((len(key_val)), True)

    for index in range(len(key_val)):
        img = tf.io.read_file(key_val[index,0])
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        #print(index)
        try:
            img = tf.image.resize(img, [MAX_HIGHT, MAX_WIDTH], preserve_aspect_ratio= True)
            clean_matrix[index] = True
        except:
            clean_matrix[index] = False
            print("removed image,    :" + str(key_val[index,0]))

    return key_val[clean_matrix]