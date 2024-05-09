from model import Finalmodel as fm
import prepare as fp
import extract as e
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model_960.py
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    plt.show()

def plot_confusion_matrix(y_true, y_pred_classes, mapping):
    cm = confusion_matrix(y_true, y_pred_classes)
    gztan = mapping
    #     "ambient",
    #     "ballad",
    #     "blues",
    #     "chillout",
    #     "classical",
    #     "country",
    #     "dance",
    #     "electronic",
    #     "folk",
    #     "funk",
    #     "jazz",
    #     "latin",
    #     "metal",
    #     "orchestral",
    #     "pop",
    #     "poprock",
    #     "reggae",
    #     "rock",
    #     "world"
    # ]
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(gztan))
    plt.xticks(tick_marks, gztan, rotation=45)
    plt.yticks(tick_marks, gztan)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    for i in range(len(gztan)):
        for j in range(len(gztan)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()

if __name__ == '__main__':

    # Initialize a list to store the scores

    data_path= r'C:\Users\david.tran\PycharmProjects\pythonProject1\JsonData\newdata_640.json'

    X, y, mapping = fp.load_data(data_path)
    # print(X.shape)

    mel_spec_train, mel_spec_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    mel_spec_train, mel_spec_validation, y_train, y_validation = train_test_split(mel_spec_train, y_train, test_size=0.2)

    mel_spec_train = mel_spec_train[..., np.newaxis]
    mel_spec_validation = mel_spec_validation[..., np.newaxis]
    mel_spec_test = mel_spec_test[..., np.newaxis]

    st = time.time()

    model = fm.build_model()
    optimiser = keras.optimizers.Adam(learning_rate=0.00012)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()


    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)

    history = model.fit(
                     mel_spec_train,
                     y_train,
                    validation_data=(
                     mel_spec_validation,
                     y_validation,),
                    batch_size=48, epochs=48, shuffle = True, callbacks=[lr_scheduler])

    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    y_pred = model.predict(mel_spec_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test

    # loaded_model = keras.saving.load_model("model.keras")

    test_loss, test_acc = model.evaluate(
         mel_spec_test,
         y_test, verbose=3)
    print('\nTest accuracy:', test_acc)


    #Demo: make a predict from audio file
    DATA_PATH = r"C:\Users\david.tran\PycharmProjects\pythonProject1\demo file\audio"
    JSON_PATH = r"C:\Users\david.tran\PycharmProjects\pythonProject1\demo file\json\demo.json"
    e.save_melspec(DATA_PATH, JSON_PATH)
    X_demo,_,u = fp.load_data(JSON_PATH)
    X_demo = X_demo[..., np.newaxis]
    y_pred_demo = model.predict(X_demo, verbose=1)
    y_pred_demo = np.argmax(y_pred_demo, axis=1)
    print("This sample seems to be of the {} genre.".format(mapping[y_pred_demo[0]]))

    # plot_history(history)
    # plot_confusion_matrix(y_test,y_pred_classes,mapping)
    # model.save("model.keras")
