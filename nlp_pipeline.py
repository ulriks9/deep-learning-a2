# Requires Microsoft Visual C++ 14.0 or higher.

from models_nlp import *
from data import *
from keras import callbacks
import matplotlib.pyplot as plt
import pickle
import time

EPOCHS = 150
INIT_LR = 1e-4
WARMUP = 0.1
L2_LAMBDA = 0.1
W_MEAN = 0
W_STD = 0.5
BATCH_SIZE = 40
DROPOUT = 0.2
PATIENCE = 2

# Can be modified to specify processes to run. Multiple of the same can be added.
# 'scratch' to train model from scratch.
# 'pre' to fine-tune pre-trained model.
# 'pred_scratch' to evaluate predictions of model from scratch.
# 'pred_pre' to evaluate predictions of pre-trained model.
routine = ['pre', 'pred_pre', 'pred_pre']

# Will overwrite previously saved objects.
save_weights = False
save_plot = True

callback = callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    patience=PATIENCE)

# Plots the history and saves the plot
def plot_history(history, filename):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1,1)

    ax.plot(epochs, acc, label="Training accuracy")
    ax.plot(epochs, val_acc, label="Validation accuracy")
    ax.legend()

    ax.set_ylim([0, 1])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")

    plt.title("Model accuracy")
    plt.savefig('plots/' + filename)
    plt.show(block=False)

def main():
    ds = DataSets()

    if not os.path.exists('datasets'):
        ds.download()

    train, val, test = ds.std_IMDB(batch_size=BATCH_SIZE)

    for model_name in routine:
        if model_name == 'scratch':
            init_weights = True
            callbacks = []
        elif model_name == 'pre':
            callbacks = [callback]
            init_weights = False
        elif model_name == 'pred_scratch':
            init_weights = True

            model = BERT(
                train_length=train.cardinality().numpy(),
                epochs=EPOCHS,
                init_weights=init_weights, 
                init_lr=INIT_LR,
                warmup = WARMUP, 
                l2_lambda=L2_LAMBDA, 
                w_mean=W_MEAN, 
                w_std=W_STD,
                dropout=DROPOUT)
            
            with open('weights/bert_scratch.pickle', 'rb') as f:
                weights = pickle.load(f)

            print("\nINFO: Performing prediction on scratch model...\n")
            model.set_weights(weights)
            result = model.evaluate(test)
            print("\nINFO: Model evaluation on test data: {}\n".format(result))
            continue
        elif model_name == 'pred_pre':
            init_weights = False

            model = BERT(
                train_length=train.cardinality().numpy(),
                epochs=EPOCHS,
                init_weights=init_weights, 
                init_lr=INIT_LR,
                warmup = WARMUP, 
                l2_lambda=L2_LAMBDA, 
                w_mean=W_MEAN, 
                w_std=W_STD,
                dropout=DROPOUT)
            
            with open('weights/bert_pre.pickle', 'rb') as f:
                weights = pickle.load(f)

            print("\nINFO: Performing prediction on pre-trained model...\n")
            model.set_weights(weights)
            result = model.evaluate(test)
            print("INFO: Model evaluation on test data: {}".format(result))
            continue

        model = BERT(
            train_length=train.cardinality().numpy(),
            epochs=EPOCHS,
            init_weights=init_weights, 
            init_lr=INIT_LR,
            warmup = WARMUP, 
            l2_lambda=L2_LAMBDA, 
            w_mean=W_MEAN, 
            w_std=W_STD,
            dropout=DROPOUT)

        print("\nINFO: Beginning training...\n")
        start = time.time()

        history = model.fit(
            x=train,
            validation_data=val,
            epochs=EPOCHS,
            callbacks=callbacks)

        end = time.time()

        print("INFO: Training took {} seconds".format(end - start))

        if save_weights:
            with open('weights/bert_{}.pickle'.format(model_name), 'wb') as f:
                pickle.dump(model.get_weights(), f)

        if save_plot:
            plot_history(history, 'bert_scratch.png')

        result = model.evaluate(test)

        print("\nINFO: Model evaluation on test data: {}\n".format(result))

main()