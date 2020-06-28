import os
from src.io import build_tfrecord_dataset
from src.evaluator import write_submission
import numpy as np
import pandas as pd
import subprocess
import re

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_probability as tfp
from tensorflow.python.feature_column.utils import sequence_length_from_sparse_tensor
from src.layer import *


# Constant
os.environ['TFHUB_CACHE_DIR'] = './tfhub/'
TSV_BASE_PATH = './data/tsv/'
TFRECORD_BASE_PATH = './data/tfrecord/'
VOCABULARY_BASE_PATH = './data/vocabulary/'


# Feature Engineering Function
def feature_engineering_fn(features):
    result = {}
    result['features'] = tf.sparse.from_dense(
        tf.io.parse_tensor(
            features['features'], out_type=tf.float32
        )
    )
    result['boxes_relative_position'] = tf.sparse.from_dense(
        tf.io.parse_tensor(
            features['boxes_relative_position'], out_type=tf.float32
        )
    )
    return result


# Input Function
parse_example_spec = {
    'image_h': tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
    'image_w': tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
    'num_boxes': tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
    'boxes_relative_size': tf.io.VarLenFeature(dtype=tf.float32),
    'query': tf.io.FixedLenFeature(shape=(1,), dtype=tf.string),
    'class_labels': tf.io.VarLenFeature(dtype=tf.string),
    'features': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'boxes_relative_position': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
}

feature_names = [
    'image_h', 'image_w', 'num_boxes',
    'boxes_relative_size',
    'query',
    'class_labels', 'features', 'boxes_relative_position'
]


def train_input_fn(batch_size, num_epochs):
    dataset = build_tfrecord_dataset(
        filenames=tf.io.gfile.glob(TFRECORD_BASE_PATH + 'train/*.tfrecord'),
        parse_example_spec=parse_example_spec,
        feature_names=feature_names,
        label_names=[],
        num_parallel_reads=8,
        shuffle=True,
        shuffle_buffer_size=1024 * 128,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_parallel_calls=4,
        prefetch_buffer_size=8,
        feature_engineering_fn=feature_engineering_fn,
    )
    return dataset


def val_input_fn(batch_size):
    dataset = build_tfrecord_dataset(
        filenames=[TFRECORD_BASE_PATH + 'valid.tfrecord'],
        parse_example_spec=parse_example_spec,
        feature_names=feature_names,
        label_names=[],
        num_parallel_reads=1,
        shuffle=False,
        shuffle_buffer_size=1024 * 8,
        num_epochs=1,
        batch_size=batch_size,
        num_parallel_calls=4,
        prefetch_buffer_size=8,
        feature_engineering_fn=feature_engineering_fn,
    )
    return dataset


def testB_input_fn(batch_size):
    dataset = build_tfrecord_dataset(
        filenames=[TFRECORD_BASE_PATH + 'testB.tfrecord'],
        parse_example_spec=parse_example_spec,
        feature_names=feature_names,
        label_names=[],
        num_parallel_reads=1,
        shuffle=False,
        shuffle_buffer_size=1024 * 8,
        num_epochs=1,
        batch_size=batch_size,
        num_parallel_calls=4,
        prefetch_buffer_size=8,
        feature_engineering_fn=feature_engineering_fn,
    )
    return dataset


# Feature Column
image_h_column = tf.feature_column.numeric_column(
    key='image_h', shape=(1, ), default_value=0.0,
    normalizer_fn=lambda x: x / 1000.0
)
image_w_column = tf.feature_column.numeric_column(
    key='image_w', shape=(1, ), default_value=0.0,
    normalizer_fn=lambda x: x / 1000.0
)
num_boxes_column = tf.feature_column.numeric_column(
    key='num_boxes', shape=(1, ), default_value=0.0,
    normalizer_fn=lambda x: x / 10.0
)

boxes_relative_size_column = tf.feature_column.sequence_numeric_column(
    key='boxes_relative_size', shape=(1,), default_value=0.0
)
boxes_relative_position_column = tf.feature_column.sequence_numeric_column(
    key='boxes_relative_position', shape=(4,), default_value=0.0
)

class_labels_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
    key='class_labels', vocabulary_file=VOCABULARY_BASE_PATH + 'class_labels'
)
class_labels_embedding_column = tf.feature_column.embedding_column(
    class_labels_column, dimension=8
)

features_column = tf.feature_column.sequence_numeric_column(
    key='features', shape=(2048,), default_value=0.0
)


# Model Function
def tfhub_embedding(x, embedding_layer, embedding_size):
    sp_tensor = tf.compat.v1.string_split(x, sep=' ')
    seq_length = sequence_length_from_sparse_tensor(sp_tensor)

    dense_tensor = tf.sparse.to_dense(sp_tensor)
    batch_size = tf.shape(dense_tensor)[0]

    flatten_embedding = embedding_layer(tf.reshape(dense_tensor, (-1,)))
    seq_embedding = tf.reshape(flatten_embedding, (batch_size, -1, embedding_size))

    return seq_embedding, seq_length


def batch_shuffle_index(params):
    batch_index = tf.range(tf.shape(params)[0])
    batch_index_shuffled = tf.random.shuffle(batch_index)
    return batch_index_shuffled


def batch_gather(params, indices):
    result = tf.gather(params, indices, axis=0)
    return result


def model_fn(features, labels, mode, params, config):
    '''
    text_feature_columns, text_feature_name
    text_pretrained_embedding_path, text_pretrained_embedding_size, text_pretrained_embedding_finetune
    image_metadata_feature_columns, image_feature_columns
    num_negative_sample
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Layers
    text_pretrained_embedding_layer = hub.KerasLayer(
        handle=params['text_pretrained_embedding_path'],
        trainable=params['text_pretrained_embedding_finetune']
    )
    text_embedding_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
    ])
    text_transformer_layer = TransformerLayer(
        attention_embedding_dim=256,
        attention_num_heads=4,
        attention_dropout=0.0,
        fnn_units=256,
        fnn_activation=tf.keras.activations.relu,
        fnn_dropout=0.1
    )
    text_transformer_norm_layer = tf.keras.layers.LayerNormalization()
    image_input_layer = tf.keras.experimental.SequenceFeatures(
        params['image_feature_columns']
    )
    image_embedding_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
    ])
    image_metadata_input_layer = tf.keras.experimental.SequenceFeatures(
        params['image_metadata_feature_columns']
    )
    image_metadata_embedding_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
    ])
    image_transformer_layer = TransformerLayer(
        attention_embedding_dim=256,
        attention_num_heads=4,
        attention_dropout=0.0,
        fnn_units=256,
        fnn_activation=tf.keras.activations.relu,
        fnn_dropout=0.1
    )
    image_transformer_norm_layer = tf.keras.layers.LayerNormalization()
    cross_attnetion_layer = CrossAttentionLayer(embedding_dim=256)
    gated_fusion_layer = GatedFusionLayer(embedding_dim=256)
    text_aggregation_layer = ScaledAttentionLayer(embedding_dim=256)
    image_aggregation_layer = ScaledAttentionLayer(embedding_dim=256)
    matching_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
    ])

    # Input
    text_input = features[params['text_feature_name']]
    image_input, image_input_sequence_length = \
        image_input_layer(features)
    image_metadata_input, image_metadata_input_sequence_length = \
        image_metadata_input_layer(features)

    # Text Pretrained Embedding
    text_pretrained_embedding, text_input_sequence_length = tfhub_embedding(
        tf.squeeze(text_input, axis=-1),
        text_pretrained_embedding_layer,
        params['text_pretrained_embedding_size']
    )

    # Text Embedding
    text_pretrained_embedding = text_embedding_model(text_pretrained_embedding)

    # Text Transformer
    text_intermediate_embedding = text_transformer_layer(
        text_pretrained_embedding,
        text_pretrained_embedding,
        text_pretrained_embedding,
        tf.sequence_mask(text_input_sequence_length),
        is_training
    )
    text_intermediate_embedding = text_transformer_norm_layer(text_intermediate_embedding)

    # Image Embedding
    image_input_embedding = image_embedding_model(image_input)

    # Image Metadata Embedding
    image_metadata_input_embedding = image_metadata_embedding_model(image_metadata_input)

    # Image Transformer
    image_intermediate_embedding = image_transformer_layer(
        image_input_embedding + image_metadata_input_embedding,
        image_input_embedding + image_metadata_input_embedding,
        image_input_embedding,
        tf.sequence_mask(image_input_sequence_length),
        is_training
    )
    image_intermediate_embedding = image_transformer_norm_layer(image_intermediate_embedding)

    # Embedding Computing Function
    def compute_text_image_embedding(
        text_intermediate_embedding,
        image_intermediate_embedding,
        text_input_sequence_length,
        image_input_sequence_length
    ):

        # Cross Attention
        text_attention_embedding, image_attention_embedding = cross_attnetion_layer(
            text_intermediate_embedding,
            image_intermediate_embedding,
            tf.sequence_mask(text_input_sequence_length),
            tf.sequence_mask(image_input_sequence_length),
            is_training
        )

        # Gated Fusion
        text_fused_embedding, image_fused_embedding = gated_fusion_layer(
            text_intermediate_embedding,
            image_intermediate_embedding,
            text_attention_embedding,
            image_attention_embedding
        )

        # Aggretation
        text_aggregated_embedding = text_aggregation_layer(
            text_fused_embedding,
            tf.sequence_mask(text_input_sequence_length)
        )
        image_aggregated_embedding = image_aggregation_layer(
            image_fused_embedding,
            tf.sequence_mask(image_input_sequence_length)
        )

        # Return
        return text_aggregated_embedding, image_aggregated_embedding

    # Text/Image Emebdding
    text_output, image_output = compute_text_image_embedding(
        text_intermediate_embedding,
        image_intermediate_embedding,
        text_input_sequence_length,
        image_input_sequence_length
    )

    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = matching_model(
            tf.concat([text_output, image_output], axis=1)
        )
        prediction = tf.nn.sigmoid(logits)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction,
            export_outputs={
                'prediction': tf.estimator.export.PredictOutput(
                    prediction
                )
            }
        )

    # EVAL / TRAIN
    logits = matching_model(
        tf.concat([text_output, image_output], axis=1)
    )
    labels = tf.ones_like(logits)

    negative_logits_list = []
    for _ in range(params['num_negative_sample']):
        shuffle_index = batch_shuffle_index(text_intermediate_embedding)
        text_output_shuffled, image_output_shuffled = compute_text_image_embedding(
            text_intermediate_embedding,
            batch_gather(image_intermediate_embedding, shuffle_index),
            text_input_sequence_length,
            batch_gather(image_input_sequence_length, shuffle_index)
        )
        negative_logits_list.append(
            matching_model(
                tf.concat([text_output_shuffled, image_output_shuffled], axis=1)
            )
        )
    negative_logits = tf.concat(negative_logits_list, axis=0)
    negative_labels = tf.zeros_like(negative_logits)

    loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(
        tf.concat([labels, negative_labels], axis=0),
        tf.concat([logits, negative_logits], axis=0)
    )
    loss = tf.math.reduce_mean(loss_vec, axis=0)

    # EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        average_loss = tf.compat.v1.metrics.mean(
            loss_vec,
            weights=tf.ones_like(loss_vec),
            name='average_loss'
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={
                'average_loss': average_loss
            }
        )

    # TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        trainable_variables = (
            text_pretrained_embedding_layer.trainable_variables +
            text_embedding_model.trainable_variables +
            text_transformer_layer.trainable_variables +
            text_transformer_norm_layer.trainable_variables +
            image_input_layer.trainable_variables +
            image_embedding_model.trainable_variables +
            image_metadata_input_layer.trainable_variables +
            image_metadata_embedding_model.trainable_variables +
            image_transformer_layer.trainable_variables +
            image_transformer_norm_layer.trainable_variables +
            cross_attnetion_layer.trainable_variables +
            gated_fusion_layer.trainable_variables +
            text_aggregation_layer.trainable_variables +
            image_aggregation_layer.trainable_variables +
            matching_model.trainable_variables
        )

        params['optimizer'].iterations = \
            tf.compat.v1.train.get_or_create_global_step()
        train_op = params['optimizer'].get_updates(
            loss=loss,
            params=trainable_variables
        )[0]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )


# Estimator
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='./estimator/',
    config=tf.estimator.RunConfig(
        tf_random_seed=0, save_summary_steps=1000, log_step_count_steps=1000,
        save_checkpoints_steps=2000, keep_checkpoint_max=20,
        session_config=tf.compat.v1.ConfigProto(device_count={'CPU': 1, 'GPU': 1})
    ),
    params={
        'text_feature_name': 'query',
        'text_pretrained_embedding_path': './tfhub/Wiki-words-250-with-normalization',
        'text_pretrained_embedding_size': 250,
        'text_pretrained_embedding_finetune': True,
        'image_metadata_feature_columns': [
            boxes_relative_size_column, boxes_relative_position_column, class_labels_embedding_column
        ],
        'image_feature_columns': [features_column],
        'num_negative_sample': 5,
        'optimizer': tf.keras.optimizers.Adam(learning_rate=3e-4)
    }
)


# Train
estimator.train(
    input_fn=lambda: train_input_fn(batch_size=256 * 1, num_epochs=1)
)

# Validation
valid_pred_iter = estimator.predict(
    input_fn=lambda: val_input_fn(batch_size=1024 * 4),
    checkpoint_path=None
)
valid_pred = [each[0] for each in valid_pred_iter]

valid_df = pd.read_csv(
    TSV_BASE_PATH + 'valid.tsv', sep='\t',
    usecols=['query_id', 'product_id']
)
valid_df['score'] = valid_pred
write_submission(valid_df, 5, './prediction/valid_pred.csv')

subprocess.call([
    'ipython', './src/evaluator.py',
    './data/tsv/valid_answer.json', './prediction/valid_pred.csv', './prediction/valid_score.json'
])


# Prediction
testB_pred_iter = estimator.predict(
    input_fn=lambda: testB_input_fn(batch_size=1024 * 4)
)
testB_pred = [each[0] for each in testB_pred_iter]

testB_df = pd.read_csv(
    TSV_BASE_PATH + 'testB.tsv', sep='\t',
    usecols=['query_id', 'product_id']
)
testB_df['score'] = testB_pred
write_submission(testB_df, 5, './prediction/testB_pred.csv')
