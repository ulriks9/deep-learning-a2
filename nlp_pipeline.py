from models_nlp import *
from data import *
from keras import callbacks
import matplotlib.pyplot as plt
import pickle
import time

EPOCHS = 150
INIT_LR = 7e-3
WARMUP = 0.1
L2_LAMBDA = 0.1
W_MEAN = 0
W_STD = 0.5
BATCH_SIZE = 50

routine = ['scratch', 'pre']
save_weights = True
save_plot = True

callback = callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    patience=10,
)

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
        else:
            callbacks=[callback]
            init_weights= False

        model = BERT(
            train_length=train.cardinality().numpy(),
            epochs=EPOCHS,
            init_weights=init_weights, 
            init_lr=INIT_LR,
            warmup = WARMUP, 
            l2_lambda=L2_LAMBDA, 
            w_mean=W_MEAN, 
            w_std=W_STD)

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

        print("INFO: Model evaluation on test data: {}".format(result))

main()