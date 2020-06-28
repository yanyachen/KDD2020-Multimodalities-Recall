import os
import csv
import urllib.request
import zipfile


data_referenhce_file = './data/raw/multimodal_data_20200605.csv'
downloading_path = './data/raw/'
tsv_path = './data/tsv/'
unzip_filenames = [
    'multimodal_train.zip',
    'multimodal_valid.zip',
    'multimodal_testA.zip',
    'multimodal_testB.zip'
]


with open(data_referenhce_file, 'r') as csv_ref:
    csv_reader = csv.DictReader(csv_ref)

    for row in csv_reader:
        if row['Area'] == 'International':
            url = row['Link']
            filename = downloading_path + os.path.basename(url)

            if not os.path.isfile(filename):
                print('Downloading:' + filename)
                urllib.request.urlretrieve(url, filename)

            if any([filename.endswith(each) for each in unzip_filenames]):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    print('Unzip:' + filename)
                    zip_ref.extractall(tsv_path)
