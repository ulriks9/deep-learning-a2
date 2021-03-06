
#mainly taken from https://www.tensorflow.org/text/tutorials/classify_text_with_bert

import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
import numpy as np

from load_data import *
from data import *

tf.get_logger().setLevel('ERROR')

INIT_RANDOMLY = True

#small bert
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
#preprocessing for small bert
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

def reset_model(model):
  weights = model.get_weights()
  print(np.shape(weights))
  weights = [np.random.normal(size=len(w.flat)).reshape(w.shape) for w in weights]
  model.set_weights(weights)

#preprocessing + bert + dropout + dense
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

#initialize
classifier_model = build_classifier_model()
if INIT_RANDOMLY:
  reset_model(classifier_model)

#train
ds = DataSets()
train_ds, test_ds = ds.IMDB(batch_size=32)
classifier_model.compile(optimizer='adam',
                         loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         metrics= tf.metrics.BinaryAccuracy())
history = classifier_model.fit(x=train_ds,validation_data=test_ds, epochs=100)

#evaluation
loss, accuracy = classifier_model.evaluate(test_ds)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

#plot progress:
history_dict = history.history

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Testing loss')
plt.title('Training and testing loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Testing acc')
plt.title('Training and testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
