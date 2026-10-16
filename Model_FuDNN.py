from keras.src.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_FuDNN(train_data, train_target, test_data, test_target, BS=None, ACT=None, SPE=None, HN=None, EP=None):

    if BS is None:
        BS = 32
    if SPE is None:
        SPE = 5
    if EP is None:
        EP = 20
    if HN is None:
        HN = 128
    if ACT is None:
        ACT = 'relu'
    input_shape = (784,)
    num_classes = train_target.shape[-1]
    SIZE = input_shape[0]

    Train_X = np.zeros((train_data.shape[0], SIZE))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], SIZE)
        Train_X[i] = temp
    Test_X = np.zeros((test_data.shape[0], SIZE))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], SIZE)
        Test_X[i] = temp

    # ---------------- Fully Connected Deep Neural Network ----------------

    model = Sequential([
        Dense(512, activation=ACT, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.40),

        Dense(256, activation=ACT),
        BatchNormalization(),
        Dropout(0.35),

        Dense(128, activation=ACT),
        BatchNormalization(),
        Dropout(0.30),

        Dense(HN, activation=ACT),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, activation=ACT),
        BatchNormalization(),
        Dropout(0.20),

        Dense(num_classes, activation='sigmoid')  # sigmoid softmax

    ])
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])  # sparse_categorical_crossentropy categorical_crossentropy
    model.fit(Train_X,train_target,epochs=EP,batch_size=BS,steps_per_epoch=SPE,validation_data=(Test_X, test_target), verbose=1)
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype(int)
    Eval = ClassificationEvaluation(test_target, pred)
    return Eval, pred

