
from pathlib import Path
import csv

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