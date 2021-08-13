import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
Kern_Path="C:/Users/user/Desktop/data/deutschl/test"
# path to json file that stores MFCCs and genre labels for each processed segment



def load_data():

    with open('input.npy', 'rb') as f:
        inputs = np.load(f)

    with open('targets.npy', 'rb') as f:
        targets = np.load(f)

    print(inputs.shape)
    print(targets.shape)
    return  inputs, targets


if __name__ == "__main__":

    # load data
    X, y = load_data()
    print(y)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



    # build network topology
    model = keras.Sequential([

        # input layer



        # 1st dense layer
        #keras.layers.Dense(512, activation='relu'),
        keras.layers.LSTM(64,input_shape=(X.shape[1], X.shape[2]),return_sequences=True),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        #keras.layers.Dense(256, activation='relu'),
        #keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(19, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()



    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20)

    model.save("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset/lstm_model.h5")

