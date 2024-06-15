import csv
import h5py
from pathlib import Path
import sys

# Siehe auch: https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn


def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(hdf5_data[obj[k][0]][0][0])
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]]])



#import digitStruct
f = h5py.File('/mnt/c/dev/extra_digitStruct.mat', 'r')
list(f.keys())
print(f.keys())
print(f['/digitStruct/name'])
size = f['/digitStruct/name'].size
print(size)

csv_data = dict()
csv_data['image_name'] = []
csv_data['label'] = []

for i in range(size):
    csv_data.update({get_name(i, f): get_box_data(i, f)['label']})
    print(i)
print(csv_data)

with open('/mnt/c/dev/extra_digitStruct.csv', "w") as f:
    w = csv.writer(f)
    for key, value in csv_data.items():
        label = ""
        for i in range(len(value)):
            label = label + str(int(value[i]))
        w.writerow([key, label])

f = h5py.File('/mnt/c/dev/extra_digitStruct.mat', 'r')
print(f)
print(get_name(1, f))
print(get_box_data(1, f)['label'])


#########
##function to create a annotation file bases on filename of an Image

# Specify the directory you want to list
directory = Path('/mnt/c/dev/Ind_vehicle_number/pics')

# Use the .glob() method to get all files (not directories)
# Verwenden Sie .glob() um alle Dateien zu bekommen (nicht Verzeichnisse)
files_only_list = [file.name for file in directory.glob('*') if file.is_file()]

print(files_only_list)

csv_file_path = '/mnt/c/dev/Ind_vehicle_number/csv_datei.csv'

# Schreiben Sie die Liste der Dateinamen in eine CSV-Datei
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for name in files_only_list:
        writer.writerow([name] + [name.split('.')])  # Jeder Dateiname in einer eigenen Zeile