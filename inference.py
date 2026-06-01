import tensorflow as tf
import numpy as np
from PIL import Image
import io
from config import CLASS_NAMES, IMG_SIZE, MODEL_PATH

# Custom classes wajib ada untuk load model
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden   = max(channels // self.ratio, 1)
        self.avg_pool      = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool      = tf.keras.layers.GlobalMaxPooling2D()
        self.shared_dense1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.shared_dense2 = tf.keras.layers.Dense(channels)

    def call(self, inputs):
        avg     = self.shared_dense2(self.shared_dense1(self.avg_pool(inputs)))
        max_val = self.shared_dense2(self.shared_dense1(self.max_pool(inputs)))
        att     = tf.nn.sigmoid(avg + max_val)
        att     = tf.reshape(att, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * att

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ratio': self.ratio})
        return cfg

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'gamma': self.gamma, 'alpha': self.alpha})
        return cfg

CUSTOM_OBJECTS = {
    'ChannelAttention': ChannelAttention,
    'FocalLoss':        FocalLoss,
}

# Load model sekali saat startup
_model = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=CUSTOM_OBJECTS
        )
        print("Model loaded!")
    return _model

def predict_image(image_bytes: bytes, top_k: int = 3) -> list:
    model = load_model()
    img   = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img   = img.resize(IMG_SIZE)
    arr   = np.array(img, dtype=np.float32)
    arr   = np.expand_dims(arr, axis=0)

    probs   = model(arr, training=False).numpy()[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    return [
        {'label': CLASS_NAMES[i], 'confidence': float(probs[i])}
        for i in top_idx
    ]