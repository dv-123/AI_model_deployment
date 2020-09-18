import numpy as np
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import save_img, load_img
import tensorflow as tf
from tensorflow.python.framework import ops

def init():
    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('model_weights.h5')
    print("Loaded Model from disk")

    loaded_model.compile(loss = 'categorical_crossentropy')
    graph = ops.get_default_graph()

    return loaded_model, graph
