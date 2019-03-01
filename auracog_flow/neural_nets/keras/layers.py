import tensorflow as tf
from keras.models import *
from keras.layers import *



class ScaledDotProductAttention(object):
    """
    Bahdanau attention model, scaled by the length of the vector (Vaswani et al. 2017).
    """
    def __init__(self, vector_size, attn_dropout=0.1):
        """
        :param vector_size:
        :param attn_dropout:
        """
        self.temper = np.sqrt(vector_size)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        """
        :param q: query
        :param k: key
        :param v: value
        :param mask: mask to be applied
        :return: (output, attn), where output = attn^T·v and attn = softmax(q^T·k / sqrt(vector_size))
        """
        # attn = q^T·k / sqrt(vector_size)
        # Dot product is made along dimension 2, i.e. q and k are supposed to be at least vectors of vectors.
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])

        # TODO: MASKING
        # print("****** mask: {}".format(mask))  # Debug
        # if mask is not None:
        #     mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
        #     attn = Add()([attn, mmask])

        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        # attn^T·v
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


# class MultiHeadSelfAttentionLayer(Layer):
#     """
#     Multihead self attention layer.
#     Input:
#         - A 3D tensor. Example: shape = (batch_size, num_features, num_words)
#     Output:
#         - A tensor with the same dimensions as input.
#     """
#     def __init__(self, n_head, d_model, d_k, d_v, dropout, use_norm=True, mask=None, **kwargs):
#         """
#         :param n_head: number of attention heads.
#         :param d_model: dimension of the features (i.e. feature vector size).
#         :param d_k: dimension of the keys in the attention model.
#         :param d_v: dimention of the values in the attention model.
#         :param dropout: (float) dropout rate.
#         :param use_norm: whether to use batch normalization.
#         """
#         self.supports_masking=True
#         super(MultiHeadSelfAttentionLayer, self).__init__(**kwargs)
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#         self.dropout = dropout
#
#         # out dimension: d_k for every head
#         self.qs_layer = Dense(n_head * d_k, use_bias=False)
#         self.ks_layer = Dense(n_head * d_k, use_bias=False)
#         # out dimension: d_v for every head
#         self.vs_layer = Dense(n_head * d_v, use_bias=False)
#
#         self.attention = ScaledDotProductAttention(d_model)
#         self.layer_norm = BatchNormalization() if use_norm else None
#
#         self.mask = mask
#
#         self.w_o = TimeDistributed(Dense(d_model))
#
#     def call(self, input, **kwargs):
#         d_k, d_v = self.d_k, self.d_v
#         n_head = self.n_head
#
#         # out dimension: d_k for every head -> n_head * d_k
#         qs = self.qs_layer(input)  # [batch_size, len_q, n_head*d_k]
#         ks = self.ks_layer(input)
#         # out dimension: d_v for every head -> n_head * d_v
#         vs = self.vs_layer(input)
#
#         def reshape1(x):
#             s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
#             # Add an extra dimension [2]: heads
#             x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
#             x = tf.transpose(x, [2, 0, 1, 3])
#             x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
#             return x
#
#         qs = Lambda(reshape1)(qs)
#         ks = Lambda(reshape1)(ks)
#         vs = Lambda(reshape1)(vs)
#
#         mask = None
#         if self.mask is not None:
#             mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(self.mask)
#         head, attn = self.attention(qs, ks, vs, mask=mask)
#
#         def reshape2(x):
#             s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
#             x = tf.reshape(x, [n_head, -1, s[1], s[2]])
#             x = tf.transpose(x, [1, 2, 0, 3])
#             x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
#             return x
#
#         head = Lambda(reshape2)(head)
#
#         outputs = self.w_o(head)
#         outputs = Dropout(self.dropout)(outputs)
#         if not self.layer_norm:
#             return outputs, attn
#         outputs = Add()([outputs, input])
#         return self.layer_norm(outputs)
#
# #    def build(self):
# #        pass


class MultiHeadSelfAttentionLayer(Layer):
    """
    Multihead self attention layer.
    Input:
        - A 3D tensor. Example: shape = (batch_size, t, features)
    Output:
        - A 3D tensor with last dimension multiplied by n_head:  (batch_size, t, features * n_head)
    """
    def __init__(self, n_head, d_model, d_k, d_v, return_attn=False, **kwargs):
        """
        :param n_head: number of attention heads.
        :param d_model: dimension of the features (i.e. feature vector size).
        :param d_k: dimension of the keys in the attention model.
        :param d_v: dimention of the values in the attention model.
        :param return_attn:
        """
        super(MultiHeadSelfAttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.return_attn = return_attn

        # # out dimension: d_k for every head
        # self.qs_layer = Dense(n_head * d_k, use_bias=False)
        # self.ks_layer = Dense(n_head * d_k, use_bias=False)
        # # out dimension: d_v for every head
        # self.vs_layer = Dense(n_head * d_v, use_bias=False)

        self.attention = ScaledDotProductAttention(d_model)
        # self.layer_norm = BatchNormalization() if use_norm else None
        #
        # self.mask = mask
        #
        # self.w_o = TimeDistributed(Dense(d_model))


    def build(self, input_shape):
        # input_shape: (batches, t, features)
        # dimensions of weight matrix: (batches, features, d_k * n_head)
        self.Wq = self.add_weight(name="Wq", shape=(input_shape[2], self.d_k * self.n_head),
                                  initializer="uniform",
                                  trainable=True)
        self.Wk = self.add_weight(name="Wk", shape=(input_shape[2], self.d_k * self.n_head),
                                  initializer="uniform",
                                  trainable=True)
        # input shape: (batches, t, features)
        # dimensions of weight matrix: (batches, features, d_v * n_head)
        self.Wv = self.add_weight(name="Wv", shape=(input_shape[2], self.d_k * self.n_head),
                                  initializer="uniform",
                                  trainable=True)
        super(MultiHeadSelfAttentionLayer, self).build(input_shape)


    def compute_output_shape(self, input_shape):
        # _res = list(input_shape)
        # _res.append(self.n_head)
        # return list(_res)
#        return (input_shape, self.n_head)
        return (input_shape[0], input_shape[1], input_shape[2]*self.n_head)


    def call(self, input, mask=None, **kwargs):
        q = K.dot(input, self.Wq)
        k = K.dot(input, self.Wk)
        v = K.dot(input, self.Wv)

        # if mask is not None:
        #     mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)

        # Apply attention
        head, attn = self.attention(q, k, v, mask=mask)

        if self.return_attn:
            return head, attn
        else:
            return head

    def get_config(self):
        config = {
            "n_head": self.n_head,
            "d_model": self.d_model,
            "d_k": self.d_k,
            "d_v": self.d_v ,
            "return_attn": self.return_attn
        }
        base_config = super(MultiHeadSelfAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



