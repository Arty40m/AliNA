import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.keras import layers as L
import numpy as np



class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_ch, drop=0., **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.c1 = L.Conv2D(out_ch, (3,3), padding='same', activation='relu',
                             kernel_initializer='he_uniform')
        self.c2 = L.Conv2D(out_ch, (3,3), padding='same', activation='relu',
                             kernel_initializer='he_uniform')
        self.c3 = L.Conv2D(out_ch, (3,3), padding='same', activation='relu',
                             kernel_initializer='he_uniform')
        self.dropout = L.Dropout(drop)

    def call(self, x):
        x1 = self.c1(x)
        x = self.c2(x1)
        x = self.dropout(x)
        x = self.c3(x)

        return x + x1


def get_pad_mask(x):

    x = tf.reshape(x, [-1, 16, 16, 256])# split rows into patches
    x = tf.transpose(x, perm=[0, 1, 3, 2])# transpose row patch
    x = tf.reshape(x, [-1, 16**2, 16**2])# (batch, seq, patch**2)

    x = tf.math.reduce_sum(x, axis=-1, keepdims=False)# (batch, seq)
    x = tf.cast((x==0.), tf.float32)# (batch, seq)

    x = tf.expand_dims(x, axis=1)# batch, 1(head), seq
    x = tf.expand_dims(x, axis=1)# batch, 1(head), 1(seq), seq

    return x


def pos_enc(max_len, d_emb):
  def _pos_enc(x):
      
      pos_enc = np.array([
      [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
      if pos != 0 else np.zeros(d_emb) 
        for pos in range(max_len)
        ])
      
      pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
      pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1

      return x+pos_enc
  return L.Lambda(_pos_enc)


class MHAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads, **kwargs):
        super(MHAttention, self).__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.norm = dim//heads

    def build(self, input_shape):        
        self.Q = self.add_weight(shape=(self.dim, self.dim),
                                 initializer='glorot_uniform', 
                                 trainable=True,
                                 name='q')
        
        self.K = self.add_weight(shape=(self.dim, self.dim),
                                 initializer='glorot_uniform', 
                                 trainable=True,
                                 name='k')

        self.V = self.add_weight(shape=(self.dim, self.dim),
                                 initializer='glorot_uniform', 
                                 trainable=True,
                                 name='v')
    
        self.O = self.add_weight(shape=(self.dim, self.dim),
                                 initializer='glorot_uniform', 
                                 trainable=True,
                                 name='o')
        
    def call(self, x):
        q, k, v, mask = x

        q = tf.linalg.matmul(q, self.Q)
        k = tf.linalg.matmul(k, self.K)
        v = tf.linalg.matmul(v, self.V)

        # batch, seq, heads, dim
        q = tf.reshape(q, (-1, q.shape[-2], self.heads, self.dim//self.heads))
        k = tf.reshape(k, (-1, k.shape[-2], self.heads, self.dim//self.heads))
        v = tf.reshape(v, (-1, v.shape[-2], self.heads, self.dim//self.heads))

        # batch, heads, seq, dim
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        #att
        g = tf.linalg.matmul(q, k, transpose_b=True)
        g /= tf.math.sqrt(float(self.norm))
        if mask is not None:
            g -= (mask*1e9)
        A = tf.nn.softmax(g, axis=-1)

        att = tf.linalg.matmul(A, v)

        att = tf.transpose(att, perm=[0, 2, 1, 3])
        att = tf.reshape(att, (-1, q.shape[-2], self.dim))
        att = tf.linalg.matmul(att, self.O)

        return att


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim, heads, do, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.do = do

        self.Att = MHAttention(self.dim, self.heads)

        self.drop1 = L.Dropout(self.do)
        self.drop2 = L.Dropout(self.do)

        self.LN1 = L.LayerNormalization(axis=-1)
        self.LN2 = L.LayerNormalization(axis=-1)

        self.FC1 = L.Dense(self.dim*4, kernel_initializer='he_uniform', 
                         activation='relu')
        self.FC2 = L.Dense(self.dim, kernel_initializer='glorot_uniform')


    def call(self, args):
        x, mask = args

        att = self.Att([x, x, x, mask])
        att = self.drop1(att)
        x = att + x
        x = self.LN1(x)

        d = self.FC1(x)
        d = self.FC2(d)
        d = self.drop2(d)

        x = d + x
        x = self.LN2(x)

        return x


def AliNA():

    Inp = L.Input(shape=(256, 256))
    mask = get_pad_mask(Inp)

    Emb = L.Embedding(821, 16, 
                  embeddings_initializer='truncated_normal')

    inp = Emb(Inp)

    # ENCODER

    x1 = ConvBlock(32, 0.2)(inp)
    x = L.MaxPool2D(2)(x1)#128

    x2 = ConvBlock(64, 0.2)(x)
    x = L.MaxPool2D(2)(x2)#64

    x3 = ConvBlock(128, 0.2)(x)
    x = L.MaxPool2D(2)(x3)#32

    x4 = ConvBlock(256, 0.2)(x)
    x = L.MaxPool2D(2)(x4)#16

    # MIDDLE

    x = L.Reshape((16**2, 256))(x)
    x = pos_enc(max_len=16**2, d_emb=256)(x)

    for l in range(3):
        x = EncoderLayer(dim=256, heads=8, do=0.1)([x, mask])

    x = L.Reshape((16, 16, 256))(x)

    # DECODER

    x = L.UpSampling2D(2, interpolation = 'bilinear')(x)#128
    x = L.Concatenate()([x4, x])
    x = ConvBlock(256, 0.2)(x)

    x = L.UpSampling2D(2, interpolation = 'bilinear')(x)#64
    x = L.Concatenate()([x3, x])
    x = ConvBlock(128, 0.2)(x)

    x = L.UpSampling2D(2, interpolation = 'bilinear')(x)#32
    x = L.Concatenate()([x2, x])
    x = ConvBlock(64, 0.2)(x)

    x = L.UpSampling2D(2, interpolation = 'bilinear')(x)#16
    x = L.Concatenate()([x1, x])
    x = ConvBlock(32, 0.1)(x)

    out = L.Conv2D(1, (3,3), padding='same', activation='sigmoid')(x)

    return Model(inputs=Inp, outputs=out)

