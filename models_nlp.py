from keras import layers, regularizers, Model, losses, metrics
from tensorflow_hub import KerasLayer
from tensorflow import string
from official.nlp import optimization
import tensorflow_text as text
import numpy as np

def BERT(train_length, reset_weights=True, epochs=100):
    # Inputs
    text_input = layers.Input(shape=(), dtype=string, name='text')
    # Pre-processor
    preprocessor = KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    # Outputs
    encoder_layer = KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2", trainable=True)
    outputs = encoder_layer(encoder_inputs)
    pooled = outputs["pooled_output"]

    # Using pooled outputs
    classifier = layers.Dropout(0.1)(pooled)
    classifier = layers.Dense(100, kernel_regularizer=regularizers.L2(0.01))(classifier)
    classifier = layers.Dense(1, activation=None)(classifier)

    # Final model
    model = Model(text_input, classifier)

    # Building optimizer
    loss = losses.BinaryCrossentropy(from_logits=True)
    metric = metrics.BinaryAccuracy()

    num_train_steps = train_length * epochs  
    num_warmup_steps = 0.1 * num_train_steps
    init_lr = 1e-4

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')

    # Compile model
    model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metric)

    if reset_weights:
        weights = model.get_weights()
        weights = [np.random.normal(loc=0, scale=0.5, size=len(w.flat)).reshape(w.shape) for w in weights]
        model.set_weights(weights)        

    return model            