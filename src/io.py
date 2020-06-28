import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import (
    NumericColumn,
    VocabularyListCategoricalColumn, VocabularyFileCategoricalColumn,
    SequenceCategoricalColumn
)
from tensorflow.python.feature_column.sequence_feature_column import (
    SequenceNumericColumn
)
import multiprocessing as mp
from collections import defaultdict, Counter


def dict_generator_to_vocabulary(
    dict_generator,
    feature_names,
    output_path
):
    vocabulary_dict = defaultdict(Counter)

    for row in dict_generator:
        for feature_name in feature_names:
            word = row[feature_name]
            vocabulary_dict[feature_name][word] += 1

    for feature_name in feature_names:
        with open(output_path + feature_name, 'w') as f:
            for word in vocabulary_dict[feature_name]:
                f.write(word + '\n')

    return None


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=map(int, value))
    )


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=map(float, value))
    )


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                bytes(str(each), encoding="utf8")
                if not isinstance(each, bytes)
                else each
                for each in value
            ]
        )
    )


def dict_to_example(
    record_dict,
    int64_cols=[],
    float_cols=[],
    bytes_cols=[]
):
    feature = dict()
    # Insert Column based on Type
    for col in int64_cols:
        if col in record_dict:
            feature[col] = _int64_feature(record_dict[col])
    for col in float_cols:
        if col in record_dict:
            feature[col] = _float_feature(record_dict[col])
    for col in bytes_cols:
        if col in record_dict:
            feature[col] = _bytes_feature(record_dict[col])
    # Generating Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def dict_generator_to_tfrecord(
    dict_generator,
    output_filename,
    int64_cols=[],
    float_cols=[],
    bytes_cols=[]
):
    # TFRecord Writer - Open
    writer = tf.io.TFRecordWriter(output_filename)
    # Reading and Writing Data
    for record_dict in dict_generator:
        example = dict_to_example(
            record_dict,
            int64_cols, float_cols, bytes_cols
        )
        writer.write(example.SerializeToString())
    # TFRecord Writer - Close
    writer.close()
    return None


def build_tfrecord_dataset(
    filenames,
    parse_example_spec,
    feature_names,
    label_names,
    num_parallel_reads,
    shuffle,
    shuffle_buffer_size,
    num_epochs,
    batch_size,
    num_parallel_calls,
    prefetch_buffer_size,
    feature_engineering_fn=None,
    **kwargs
):
    # Reading
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=num_parallel_reads
    )

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=0, reshuffle_each_iteration=True
        )
    # Repeat
    if num_epochs > 1:
        dataset = dataset.repeat(num_epochs)

    # Mapping Function
    def parse_fn(serialized):
        parsed_example = tf.io.parse_single_example(
            serialized, parse_example_spec
        )
        features = {each: parsed_example[each] for each in feature_names}
        if feature_engineering_fn:
            features_new = feature_engineering_fn(features, **kwargs)
            features.update(features_new)
        if len(label_names) == 0:
            return features
        elif len(label_names) == 1:
            labels = parsed_example[label_names[0]]
        else:
            labels = {each: parsed_example[each] for each in label_names}
        return features, labels

    # Mapping
    dataset = dataset.map(parse_fn, num_parallel_calls)

    # Batch
    dataset = dataset.batch(batch_size)

    # Prefetch
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset


def build_parse_example_spec(feature_columns):
    result = {}
    for feature_column in feature_columns:
        if isinstance(
            feature_column,
            (NumericColumn, SequenceNumericColumn)
        ):
            result.update(feature_column.parse_example_spec)
        elif isinstance(
            feature_column,
            (VocabularyListCategoricalColumn, VocabularyFileCategoricalColumn)
        ):
            result.update(
                {
                    feature_column.key: tf.io.FixedLenFeature(
                        shape=(1,),
                        dtype=feature_column.dtype,
                        default_value=''
                    )
                }
            )
        elif isinstance(feature_column, SequenceCategoricalColumn):
            result.update(feature_column.parse_example_spec)
        else:
            ValueError('Invalid Feature Column:', feature_column)
    return result
