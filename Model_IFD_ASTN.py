import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation


# Spatial Transformer Layer

class SpatialTransformer(layers.Layer):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.localization = models.Sequential([
            layers.Conv2D(8, (7,7), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(10, (5,5), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(
                6,
                kernel_initializer='zeros',
                bias_initializer=tf.constant_initializer(
                    [1,0,0,0,1,0]
                )
            )
        ])

    def call(self, x):
        theta = self.localization(x)
        theta = tf.reshape(
            theta,
            (-1,2,3)
        )
        return x


def Model_IFD_ASTN(train_data, train_target, test_data, test_target,
                   SPE=None, EP=None, HN=None, ACT=None, BS=None, sol=None):

    if sol is None:
        sol = [128, 0.001, 100]
    if SPE is None:
        SPE = int(sol[2])  # 5
    if EP is None:
        EP = 30
    if HN is None:
        HN = int(sol[0])  # 128
    if BS is None:
        BS = 32
    if ACT is None:
        ACT = 'relu'
    LR = sol[1]  # 0.001
    IMG_SIZE = 32

    # Image Reshape
    Train_X = np.zeros((train_data.shape[0],IMG_SIZE,IMG_SIZE,3))

    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE*IMG_SIZE,3))
        Train_X[i] = temp.reshape(IMG_SIZE,IMG_SIZE,3)

    Test_X = np.zeros((test_data.shape[0],IMG_SIZE,IMG_SIZE,3))

    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i],(IMG_SIZE*IMG_SIZE,3))
        Test_X[i] = temp.reshape(IMG_SIZE,IMG_SIZE,3)
    classes = train_target.shape[-1]

    # ASTN + CNN Network
    inputs = layers.Input(shape=(32,32,3))

    # Adaptive Spatial Transformer
    x = SpatialTransformer()(inputs)

    # ---------------- CNN Backbone ----------------
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    intermediate_feature1 = x
    x = layers.Conv2D(64, (3,3), activation='relu')(x)

    x = layers.MaxPooling2D((2,2))(x)
    intermediate_feature2 = x
    x = layers.Conv2D(64,(3,3),activation='relu')(x)
    x = layers.Flatten()(x)

    # Feature representation
    feature_vector = layers.Dense(HN, activation=ACT, name="Intermediate_Feature")(x)  # 'relu'
    output = layers.Dense(classes,activation='softmax')(feature_vector)
    model = models.Model(inputs, output)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, train_target, epochs=EP, batch_size=BS, steps_per_epoch=SPE, validation_data=(Test_X, test_target), verbose=1)
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(test_target, pred)
    return Eval, pred


