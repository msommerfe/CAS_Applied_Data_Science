
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

## Reads the Images and labels out of a parquet file and writes it in a Annotation csv an stores all images as well


new_parquet_df = pd.read_parquet('/mnt/c/dev/tmp/train-00000-of-00001.parquet')
print(new_parquet_df['image'][0])

images = new_parquet_df['image']
labels = new_parquet_df['label']

i = 0
filenames = []
annotations = []
for image in images:
    binary_data = image['bytes']
    filename = os.path.join('/mnt/c/dev/tmp/', str(i) + '.png')
    filenames.append(str(i) + '.png')
    i = i+1
    with open(filename, 'wb') as file:
        file.write(binary_data)

    print(f"Image saved as {filename}")

for label in labels:
    annotations.append(label)



with open(os.path.join('/mnt/c/dev/tmp/','annotations.csv'), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter= ' ')
    for n in range(len(filenames)):
        writer.writerow([str(filenames[n]), str(annotations[n]), ''])












'''



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
        
'''