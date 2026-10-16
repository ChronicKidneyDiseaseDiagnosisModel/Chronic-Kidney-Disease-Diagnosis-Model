from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import cv2 as cv
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_XceptionNet(Train_Data, Train_Target, Test_Data, Test_Target,
                      EP=None, ACT=None, BS=None):
    if BS is None:
        BS = 4
    if EP is None:
        EP = 10
    if ACT is None:
        ACT = 'relu'
    input_shape = (299, 299, 3)  # Xception requires input size of (299, 299)
    num_classes = Test_Target.shape[-1]

    train_data = np.zeros((Train_Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (input_shape[0], input_shape[1], input_shape[2]))
        train_data[i] = temp

    test_data = np.zeros((Test_Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (input_shape[0], input_shape[1], input_shape[2]))
        test_data[i] = temp

    # Load the Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=ACT)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of output classes

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train_data, Train_Target, epochs=EP, batch_size=BS, steps_per_epoch=5, validation_data=(test_data, Test_Target), verbose=1)
    pred = model.predict(test_data)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(Test_Target, pred)
    return Eval, pred


