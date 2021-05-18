import tensorflow as tf
  
tf.set_random_seed(0)

class MultiHeadedAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads=1):
    super(MultiHeadedAttention, self).__init__()
    self.d_model = d_model
    self.d_k = self.d_model // num_heads
    self.sqrt_d_k = tf.sqrt(tf.cast(self.d_k, tf.float32))
    self.num_heads = num_heads
    self.attns = None
    self._make_linear_projections()

  def _make_linear_projections(self):
    self.proj_q = tf.keras.layers.Dense(self.d_model)
    self.proj_k = tf.keras.layers.Dense(self.d_model)
    self.proj_v = tf.keras.layers.Dense(self.d_model)
    self.proj_o = tf.keras.layers.Dense(self.d_model)

  def _attention(self, query, key, value, linear, scale):
    if linear:
        query = self.proj_q(query)
        key = self.proj_q(key)
        value = self.proj_v(value)

    if scale:
        logits = tf.matmul(query, key, transpose_b=True) / self.sqrt_d_k
    else:
        logits = tf.matmul(query, key, transpose_b=True) # / self.d_k 

    scores = tf.nn.softmax(logits, -1)

    return tf.matmul(scores, value), scores

  def call(self, query, key, value, linear=False, scale=False):
    scores, self.attns = self._attention(query, key, value, linear=linear, scale=scale)
    scores = tf.reshape(scores, [scores.shape[0], -1, self.d_model])
    if linear:
        scores = self.proj_o(scores)
    return scores, self.attns
