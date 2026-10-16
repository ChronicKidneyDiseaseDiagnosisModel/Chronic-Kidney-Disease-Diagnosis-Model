import numpy as np
from keras import layers, models
from keras.optimizers import Adam
from sklearn.svm import SVC
from tensorflow.keras.models import Model
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_CNN_SVM(train_data, train_target, test_data, test_target,
                  SPE=None, EP=None, HN=None, BS=None, ACT=None, sol=None):
    if BS is None:
        BS = 4
    if EP is None:
        EP = 30
    if ACT is None:
        ACT = 'relu'
    if SPE is None:
        SPE = 5
    if HN is None:
        HN = 128

    IMG_SIZE = 32

    # -------- Image Resizing --------
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = temp.reshape(IMG_SIZE, IMG_SIZE, 3)

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = temp.reshape(IMG_SIZE, IMG_SIZE, 3)

    # Convert one-hot labels to class labels
    if train_target.shape[-1] == 1:
        y_train = np.reshape(train_target, (-1))
        y_test = np.reshape(test_target, (-1))
        Test_Targets = np.zeros((test_target.shape[0], 2), dtype=int)
        Test_Targets[np.arange(test_target.shape[0]), test_target.flatten()] = 1
    else:
        y_train = np.argmax(train_target, axis=1)
        y_test = np.argmax(test_target, axis=1)
        Test_Targets = test_target

    inputs = layers.Input( shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation=ACT)(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=ACT)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=ACT)(x)
    x = layers.Flatten()(x)
    feature_output = layers.Dense(HN, activation=ACT, name="FeatureLayer")(x)
    outputs = layers.Dense(train_target.shape[1], activation='softmax')(feature_output)
    cnn = Model(inputs=inputs, outputs=outputs)
    cnn.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(Train_X, train_target, epochs=EP, batch_size=BS, steps_per_epoch=SPE, verbose=1)

    feature_model = Model(inputs=cnn.input, outputs=cnn.get_layer("FeatureLayer").output)
    Train_Features = feature_model.predict(Train_X)
    Test_Features = feature_model.predict(Test_X)
    svm_model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    svm_model.fit(Train_Features, y_train)
    pred = svm_model.predict_proba(Test_Features)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    if train_target.shape[-1] == 1:
        pred_label = np.argmax(pred, axis=1)
    else:
        pred_label = pred
    Eval = ClassificationEvaluation(Test_Targets, pred)
    return Eval, pred_label

