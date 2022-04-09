from models_nlp import *
from data import *
import matplotlib.pyplot as plt
import pickle

EPOCHS = 125

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
    plt.show()

def main():
    ds = DataSets()

    # UNCOMMENT THIS LINE AFTER FIRST RUN
    ds.download()

    train, val, test = ds.std_IMDB(batch_size=30)
    # model = BERT(train_length=train.cardinality().numpy(), epochs=EPOCHS, reset_weights=False)

    # history = model.fit(x=train,
    #                 validation_data=val,
    #                 epochs=EPOCHS)
    
    # with open('weights/bert_pre.pickle', 'wb') as f:
    #     pickle.dump(model.get_weights(), f)

    # plot_history(history, 'bert_pre_trained.png')

    model = BERT(train_length=train.cardinality().numpy(), epochs=EPOCHS, reset_weights=True)

    history = model.fit(x=train,
                    validation_data=val,
                    epochs=EPOCHS)

    with open('weights/bert_scratch.pickle', 'wb') as f:
        pickle.dump(model.get_weights(), f)

    plot_history(history, 'bert_scratch.png')                    

main()