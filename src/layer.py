import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_file


class VocabularyFileIndexLayer(tf.keras.layers.Layer):

    def __init__(self, vocabulary_file, num_oov_buckets, **kwargs):
        super(VocabularyFileIndexLayer, self).__init__(**kwargs)
        self.vocabulary_file = vocabulary_file
        self.num_oov_buckets = num_oov_buckets

    def build(self, input_shape):
        super(VocabularyFileIndexLayer, self).build(input_shape)
        self.table = index_table_from_file(
            vocabulary_file=self.vocabulary_file,
            num_oov_buckets=self.num_oov_buckets,
        )

    def call(self, inputs):
        outputs = self.table.lookup(inputs)
        return outputs

    def get_config(self):
        config = super(VocabularyFileIndexLayer, self).get_config()
        config.update({
            'vocabulary_file': self.vocabulary_file,
            'num_oov_buckets': self.num_oov_buckets
        })
        return config


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(ScaledDotProductAttentionLayer, self).__init__()

    def call(
        self,
        query,
        key,
        value,
        query_mask,
        key_mask
    ):
        # Prep
        embedding_size = tf.cast(tf.shape(value)[2], tf.float32)

        # Attnetion
        attention_weights = tf.keras.backend.batch_dot(
            query, key,
            axes=[2, 2]
        )

        # Scale
        attention_weights_scaled = attention_weights / tf.sqrt(embedding_size)

        # Mask
        query_mask01 = tf.cast(query_mask, tf.float32)
        key_mask01 = tf.cast(key_mask, tf.float32)
        attention_mask = tf.cast(
            tf.einsum('nu, nv -> nuv', query_mask01, key_mask01),
            tf.bool
        )
        attention_weights_scaled_masked = tf.where(
            attention_mask,
            attention_weights_scaled,
            tf.ones_like(attention_weights_scaled) * tf.pow(-2.0, 31)
        )

        # Softmax
        attention_softmax = tf.nn.softmax(
            attention_weights_scaled_masked,
            axis=-1
        )
        attention_softmax = tf.where(
            attention_mask,
            attention_softmax,
            tf.zeros_like(attention_softmax)
        )

        # Attention
        attention_logits = tf.keras.backend.batch_dot(
            attention_softmax, value,
            axes=[2, 1]
        )
        return attention_logits


class CrossAttentionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        embedding_dim
    ):
        super(CrossAttentionLayer, self).__init__()

        self.embedding_dim = embedding_dim

        self.text_projection_layer = tf.keras.layers.Dense(self.embedding_dim)
        self.image_projection_layer = tf.keras.layers.Dense(self.embedding_dim)

        self.text_attention_layer = MultiHeadAttentionLayer(self.embedding_dim, 4, 0.0)
        self.image_attention_layer = MultiHeadAttentionLayer(self.embedding_dim, 4, 0.0)

        self.text_attention_norm_layer = tf.keras.layers.LayerNormalization()
        self.image_attention_norm_layer = tf.keras.layers.LayerNormalization()

        self.text_self_attention_layer = MultiHeadAttentionLayer(self.embedding_dim, 4, 0.0)
        self.image_self_attention_layer = MultiHeadAttentionLayer(self.embedding_dim, 4, 0.0)

        self.text_self_attention_norm_layer = tf.keras.layers.LayerNormalization()
        self.image_self_attention_norm_layer = tf.keras.layers.LayerNormalization()

    def call(
        self,
        text_embedding,
        image_embedding,
        text_mask,
        image_mask,
        training
    ):
        # Linear Projections
        text_projected_embedding = self.text_projection_layer(text_embedding)
        image_projected_embedding = self.image_projection_layer(image_embedding)

        # Cross Attention
        text_attention_embedding = self.text_attention_layer(
            image_projected_embedding,
            text_projected_embedding,
            text_embedding,
            image_mask,
            text_mask
        )
        image_attention_embedding = self.image_attention_layer(
            text_projected_embedding,
            image_projected_embedding,
            image_embedding,
            text_mask,
            image_mask
        )
        text_attention_embedding = self.text_attention_norm_layer(text_attention_embedding)
        image_attention_embedding = self.image_attention_norm_layer(image_attention_embedding)

        # Self Attention
        text_attention_embedding = self.text_self_attention_layer(
            text_attention_embedding,
            text_attention_embedding,
            text_attention_embedding,
            image_mask,
            image_mask,
            training
        )
        image_attention_embedding = self.image_self_attention_layer(
            image_attention_embedding,
            image_attention_embedding,
            image_attention_embedding,
            text_mask,
            text_mask,
            training
        )
        text_attention_embedding = self.text_self_attention_norm_layer(text_attention_embedding)
        image_attention_embedding = self.image_self_attention_norm_layer(image_attention_embedding)

        return text_attention_embedding, image_attention_embedding


class GatedFusionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        embedding_dim
    ):
        super(GatedFusionLayer, self).__init__()

        self.embedding_dim = embedding_dim

        self.text_projection_layer = tf.keras.layers.Dense(self.embedding_dim)
        self.image_projection_layer = tf.keras.layers.Dense(self.embedding_dim)

    def call(
        self,
        text_embedding,
        image_embedding,
        text_attention_embedding,
        image_attention_embedding
    ):
        # Text
        text_fusion_gate = tf.keras.activations.hard_sigmoid(
            tf.reduce_sum(
                text_embedding * image_attention_embedding,
                axis=-1,
                keepdims=True
            )
        )
        text_fused_embedding = self.text_projection_layer(
            tf.multiply(
                text_fusion_gate,
                text_embedding + image_attention_embedding
            )
        ) + text_embedding

        # Image
        image_fusion_gate = tf.keras.activations.hard_sigmoid(
            tf.reduce_sum(
                image_embedding * text_attention_embedding,
                axis=-1,
                keepdims=True
            )
        )
        image_fused_embedding = self.image_projection_layer(
            tf.multiply(
                image_fusion_gate,
                image_embedding + text_attention_embedding
            )
        ) + image_embedding

        return text_fused_embedding, image_fused_embedding


class ScaledAttentionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        embedding_dim
    ):
        super(ScaledAttentionLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.weight = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform(0)(shape=(embedding_dim, 1)),
            trainable=True
        )

    def call(
        self,
        inputs,
        mask
    ):
        # Attention Weights
        attention_weights = tf.squeeze(tf.linalg.matmul(inputs, self.weight), -1)

        # Scale
        attention_weights_scaled = attention_weights / tf.sqrt(float(self.embedding_dim))

        # Mask
        attention_weights_scaled_masked = tf.where(
            mask,
            attention_weights_scaled,
            tf.ones_like(attention_weights_scaled) * tf.pow(-2.0, 31)
        )

        # Softmax
        attention_softmax = tf.nn.softmax(attention_weights_scaled_masked, -1)

        # Attention
        attention_logits = tf.keras.backend.batch_dot(
            attention_softmax, inputs,
            axes=[1, 1]
        )

        return attention_logits


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout,
        use_bias=False
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias

        self.query_projection_layer = tf.keras.layers.Dense(
            units=self.embedding_dim,
            activation=tf.keras.activations.linear,
            use_bias=self.use_bias
        )
        self.key_projection_layer = tf.keras.layers.Dense(
            units=self.embedding_dim,
            activation=tf.keras.activations.linear,
            use_bias=self.use_bias
        )
        self.value_projection_layer = tf.keras.layers.Dense(
            units=self.embedding_dim,
            activation=tf.keras.activations.linear,
            use_bias=self.use_bias
        )
        self.output_projection_layer = tf.keras.layers.Dense(
            units=self.embedding_dim,
            activation=tf.keras.activations.linear,
            use_bias=self.use_bias
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=float(dropout))

    def call(
        self,
        query,
        key,
        value,
        query_mask,
        key_mask,
        training
    ):
        # QKV Linear Projections
        query = self.query_projection_layer(query)
        key = self.key_projection_layer(key)
        value = self.value_projection_layer(value)

        # Multi-Head Split
        query = tf.concat(tf.split(query, self.num_heads, axis=2), axis=0)
        key = tf.concat(tf.split(key, self.num_heads, axis=2), axis=0)
        value = tf.concat(tf.split(value, self.num_heads, axis=2), axis=0)

        # Dot Product
        attention_weights = tf.keras.backend.batch_dot(
            query, key,
            axes=[2, 2]
        )

        # Scale
        depth = (self.embedding_dim // self.num_heads)
        attention_weights_scaled = attention_weights / tf.sqrt(float(depth))

        # Mask
        query_multihead_mask01 = tf.tile(
            tf.cast(query_mask, tf.float32),
            (self.num_heads, 1)
        )
        key_multihead_mask01 = tf.tile(
            tf.cast(key_mask, tf.float32),
            (self.num_heads, 1)
        )
        multihead_attention_mask = tf.cast(
            tf.einsum('nu, nv -> nuv', query_multihead_mask01, key_multihead_mask01),
            tf.bool
        )
        attention_weights_scaled_masked = tf.where(
            multihead_attention_mask,
            attention_weights_scaled,
            tf.ones_like(attention_weights_scaled) * tf.pow(-2.0, 31)
        )

        # Softmax
        attention_softmax = tf.nn.softmax(
            attention_weights_scaled_masked,
            axis=-1
        )
        attention_softmax = tf.where(
            multihead_attention_mask,
            attention_softmax,
            tf.zeros_like(attention_softmax)
        )

        # Dropout
        attention_softmax = self.dropout_layer(
            attention_softmax, training=training
        )

        # Attention
        attention_logits = tf.keras.backend.batch_dot(
            attention_softmax, value,
            axes=[2, 1]
        )

        # Multi-Head Combine
        attention_logits = tf.concat(
            tf.split(attention_logits, self.num_heads, axis=0), axis=2
        )

        # Output Linear Projection
        attention_logits = self.output_projection_layer(attention_logits)

        return attention_logits


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        attention_embedding_dim,
        attention_num_heads,
        attention_dropout,
        fnn_units,
        fnn_activation,
        fnn_dropout
    ):
        super(TransformerLayer, self).__init__()
        self.attention_embedding_dim = attention_embedding_dim
        self.attention_num_heads = attention_num_heads
        self.attention_dropout = attention_dropout
        self.fnn_units = fnn_units
        self.fnn_activation = fnn_activation
        self.fnn_dropout = fnn_dropout

        self.attention_layer = MultiHeadAttentionLayer(
            self.attention_embedding_dim,
            self.attention_num_heads,
            self.attention_dropout
        )
        self.attention_norm_layer = tf.keras.layers.LayerNormalization()
        self.fnn_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=fnn_units, activation=fnn_activation),
            tf.keras.layers.Dropout(fnn_dropout)
        ])
        self.fnn_norm_layer = tf.keras.layers.LayerNormalization()

    def call(
        self,
        query,
        key,
        value,
        mask,
        training
    ):
        # Attention
        attention_logits = self.attention_layer(
            query, key, value,
            mask, mask,
            training
        )
        attention_logits = attention_logits + value
        attention_logits = self.attention_norm_layer(attention_logits)

        # FFN
        ffn_logits = self.fnn_layer(attention_logits)
        ffn_logits = ffn_logits + attention_logits
        ffn_logits = self.fnn_norm_layer(ffn_logits)

        return ffn_logits


class InnerProductLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(InnerProductLayer, self).__init__()

    def call(self, inputs):
        outputs = tf.reduce_sum(
            tf.math.multiply(inputs[0], inputs[1]),
            axis=1
        )
        return outputs
