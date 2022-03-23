#follows the tutorial on: https://www.tensorflow.org/text/tutorials/classify_text_with_bert

from asyncio.windows_events import NULL
from tensorflow.keras.utils import text_dataset_from_directory
from tensorflow.data import AUTOTUNE

DATA_PATH = "small_aclImdb"

def load_data(batch_size = None):
    raw_train_ds = text_dataset_from_directory(
        DATA_PATH + '/train',
        batch_size = batch_size )
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = text_dataset_from_directory(
        DATA_PATH +'/test',
        batch_size= batch_size )
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds

    #class_names = raw_train_ds.class_names
    #for text_batch, label_batch in train_ds:
    #    print(f'Review: {text_batch.numpy()}')
    #    label = label_batch.numpy()
    #    print(f'Label : {label} ({class_names[label]})')

    #for text_batch, label_batch in train_ds.take(1):
    #    for i in range(3):
    #        print(f'Review: {text_batch.numpy()[i]}')
    #        label = label_batch.numpy()[i]
    #        print(f'Label : {label} ({class_names[label]})')

