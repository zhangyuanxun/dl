import tensorflow as tf


class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, agg_mode='sum', **kwargs):
        super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)

        if agg_mode not in ['sum', 'mean']:
            raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
        self.agg_mode = agg_mode

    @tf.function
    def call(self, inputs, mask=None):
        masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
        if self.agg_mode == 'sum':
            aggregated = tf.reduce_sum(masked_embeddings, axis=1)
        elif self.agg_mode == 'mean':
            aggregated = tf.reduce_mean(masked_embeddings, axis=1)

        return aggregated

    def get_config(self):
        # this is used when loading a saved model that uses a custom layer
        return {'agg_mode': self.agg_mode}


class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask


class YoutubeDNN(object):
    def __init__(self, user_feature_columns, item_feature_columns, num_sample=5,
                 hidden_units=(64, 32), activation='relu', use_bn=False, l2_reg=0.0,
                 l2_reg_embedding=1e-6, dropout=0, init_std=0.0001, seed=1024):

        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.num_sample = num_sample
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bn = use_bn
        self.l2_reg = l2_reg
        self.l2_reg_embedding = l2_reg_embedding

    def create_model(self):
        item_feature_name = self.item_feature_columns[0].name
        item_vocabulary_size = self.item_feature_columns[0].vocabulary_size

