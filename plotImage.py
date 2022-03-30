import pickle

def plot_data(preaccuracy, ptaccuracy, accuracy):
    plt.plot(preaccuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../preaccuracy.jpg')
    plt.clf()

    plt.plot(ptaccuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../paccuracy.jpg')
    plt.clf()

    plt.plot(accuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../accuracy.jpg')
    plt.clf()
    
    plt.plot(ptaccuracy, 'b')
    plt.plot(accuracy, 'g')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../both_accuracy.jpg')
    plt.clf()

def main():

    
    with open('prehistory.history', 'rb') as config_history_file:
        prehistory = pickle.load(config_history_file)

    with open('pthistory.history', 'rb') as config_history_file:
        pthistory = pickle.load(config_history_file)

    with open('history.history', 'rb') as config_history_file:
        history = pickle.load(config_history_file)

    plot_data(prehistory.history['accuracy'], pthistory.history['accuracy'], history.history['accuracy'])
    


if __name__ == "__main__":
    main()