from tqdm import tqdm
import base64
import sys
import csv
import numpy as np
import tensorflow as tf
from src.io import dict_generator_to_tfrecord
from collections import Counter

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

# Constant
TSV_BASE_PATH = './data/tsv/'
TFRECORD_BASE_PATH = './data/tfrecord/'
VOCABULARY_BASE_PATH = './data/vocabulary/'


# Feature Extraction Function
def convert_boxes(boxes, num_boxes):
    return np.frombuffer(
        base64.b64decode(boxes), dtype=np.float32
    ).reshape(int(num_boxes), 4)


def convert_features(features, num_boxes):
    return np.frombuffer(
        base64.b64decode(features), dtype=np.float32
    ).reshape(int(num_boxes), 2048)


def convert_class_labels(class_labels, num_boxes):
    return np.frombuffer(
        base64.b64decode(class_labels), dtype=np.int64
    ).reshape(int(num_boxes))


def convert_relative_position(boxes, image_h, image_w):
    return np.concatenate(
        [
            boxes[:, [0, 2]] / image_h,
            boxes[:, [1, 3]] / image_w
        ],
        axis=1
    )


def convert_relative_size(boxes, image_h, image_w):
    return np.divide(
        np.multiply(
            (boxes[:, 2] - boxes[:, 0]),
            (boxes[:, 3] - boxes[:, 1])
        ),
        (image_h * image_w)
    )


class TextPreProcessor:

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):
        word_list = self.tokenizer.tokenize(text)
        word_list = [
            self.lemmatizer.lemmatize(word)
            for word in word_list
            if word not in self.stopwords_set
        ]
        result = ' '.join(word_list)
        return result


text_preprocessor = TextPreProcessor()


# DataSet Generator
def dataset_generator(input_filename, num_file, file_id):

    with open(input_filename, 'r') as f:

        csv_reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for i, row in enumerate(csv_reader):

            if i % num_file != file_id:
                continue

            result = dict()

            # Raw
            # FixedLenFeature
            result['product_id'] = str(row['product_id'])
            result['image_h'] = int(row['image_h'])
            result['image_w'] = int(row['image_w'])
            result['num_boxes'] = int(row['num_boxes'])
            result['query'] = str(text_preprocessor(row['query']))
            result['query_id'] = str(row['query_id'])

            # Raw
            # VarLenFeature
            result['class_labels'] = convert_class_labels(
                row['class_labels'], row['num_boxes']
            )
            # VarLenFeature + Non Scalar
            result['features'] = convert_features(
                row['features'], row['num_boxes']
            )

            # Engineering
            boxes = convert_boxes(row['boxes'], row['num_boxes'])
            # VarLenFeature
            result['boxes_relative_size'] = convert_relative_size(
                boxes, result['image_h'], result['image_w']
            )
            # VarLenFeature + Non Scalar
            result['boxes_relative_position'] = convert_relative_position(
                boxes, result['image_h'], result['image_w']
            )

            # Tensor Serialization
            result['class_labels'] = result['class_labels'].tolist()
            result['features'] = tf.io.serialize_tensor(result['features']).numpy()
            result['boxes_relative_size'] = result['boxes_relative_size'].tolist()
            result['boxes_relative_position'] = tf.io.serialize_tensor(result['boxes_relative_position']).numpy()

            # Return
            yield result


# Write TFRecord
csv.field_size_limit(sys.maxsize)
FLOAT_COLS = ['image_h', 'image_w', 'num_boxes', 'boxes_relative_size']
BYTES_COLS = [
    'product_id', 'query', 'query_id', 'class_labels',
    'features', 'boxes_relative_position'
]
NUM_FILE = 32

# Train Data

# 1
for i in tqdm(range(0, (NUM_FILE // 4) * 1, +1)):
    train_generator = dataset_generator(
        TSV_BASE_PATH + 'train.tsv', NUM_FILE, i
    )

    dict_generator_to_tfrecord(
        dict_generator=train_generator,
        output_filename=TFRECORD_BASE_PATH + 'train/train_{id}.tfrecord'.format(id=i),
        int64_cols=[],
        float_cols=FLOAT_COLS,
        bytes_cols=BYTES_COLS
    )
# 2
for i in tqdm(range((NUM_FILE // 4) * 1, (NUM_FILE // 4) * 2, +1)):
    train_generator = dataset_generator(
        TSV_BASE_PATH + 'train.tsv', NUM_FILE, i
    )

    dict_generator_to_tfrecord(
        dict_generator=train_generator,
        output_filename=TFRECORD_BASE_PATH + 'train/train_{id}.tfrecord'.format(id=i),
        int64_cols=[],
        float_cols=FLOAT_COLS,
        bytes_cols=BYTES_COLS
    )
# 3
for i in tqdm(range((NUM_FILE // 4) * 2, (NUM_FILE // 4) * 3, +1)):
    train_generator = dataset_generator(
        TSV_BASE_PATH + 'train.tsv', NUM_FILE, i
    )

    dict_generator_to_tfrecord(
        dict_generator=train_generator,
        output_filename=TFRECORD_BASE_PATH + 'train/train_{id}.tfrecord'.format(id=i),
        int64_cols=[],
        float_cols=FLOAT_COLS,
        bytes_cols=BYTES_COLS
    )
# 4
for i in tqdm(range((NUM_FILE // 4) * 3, NUM_FILE, +1)):
    train_generator = dataset_generator(
        TSV_BASE_PATH + 'train.tsv', NUM_FILE, i
    )

    dict_generator_to_tfrecord(
        dict_generator=train_generator,
        output_filename=TFRECORD_BASE_PATH + 'train/train_{id}.tfrecord'.format(id=i),
        int64_cols=[],
        float_cols=FLOAT_COLS,
        bytes_cols=BYTES_COLS
    )

# Validation Data
validation_generator = dataset_generator(
    TSV_BASE_PATH + 'valid.tsv', 1, 0
)

dict_generator_to_tfrecord(
    dict_generator=validation_generator,
    output_filename=TFRECORD_BASE_PATH + 'valid.tfrecord',
    int64_cols=[],
    float_cols=FLOAT_COLS,
    bytes_cols=BYTES_COLS
)

# Test A Data
testA_generator = dataset_generator(
    TSV_BASE_PATH + 'testA.tsv', 1, 0
)

dict_generator_to_tfrecord(
    dict_generator=testA_generator,
    output_filename=TFRECORD_BASE_PATH + 'testA.tfrecord',
    int64_cols=[],
    float_cols=FLOAT_COLS,
    bytes_cols=BYTES_COLS
)

# Test B Data
testB_generator = dataset_generator(
    TSV_BASE_PATH + 'testB.tsv', 1, 0
)

dict_generator_to_tfrecord(
    dict_generator=testB_generator,
    output_filename=TFRECORD_BASE_PATH + 'testB.tfrecord',
    int64_cols=[],
    float_cols=FLOAT_COLS,
    bytes_cols=BYTES_COLS
)
