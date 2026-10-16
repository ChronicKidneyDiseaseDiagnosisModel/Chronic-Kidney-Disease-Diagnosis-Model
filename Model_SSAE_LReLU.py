import keras
import numpy as np
from keras import layers, regularizers, Model
from sklearn.model_selection import train_test_split


def Model_SSAE_LReLU(Data, Targets):
    train_data, test_data, train_target, test_target = train_test_split(
        Data,
        Targets,
        test_size=0.25,
        random_state=104,
        shuffle=True
    )
    input_dim = train_data.shape[1]
    # Input Layer
    input_layer = layers.Input(shape=(input_dim,))
    # Encoder (Stacked Sparse Autoencoder)
    x = layers.Dense(512, activity_regularizer=regularizers.l1(1e-5))(input_layer)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(256,activity_regularizer=regularizers.l1(1e-5))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(128, activity_regularizer=regularizers.l1(1e-5))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Bottleneck Feature Layer
    encoded = layers.Dense(64, activity_regularizer=regularizers.l1(1e-5), name="FeatureLayer")(x)
    encoded = layers.LeakyReLU(alpha=0.1)(encoded)
    # Decoder
    x = layers.Dense(128)(encoded)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True, validation_data=(test_data, test_data), verbose=1)
    feature_model = Model(inputs=model.input, outputs=model.get_layer("FeatureLayer").output)
    predict = feature_model.predict(Data)
    pred = model.predict(test_data)

    layerNo = -5
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layerNo].output)
    Feats = intermediate_model.predict(np.concatenate((train_data, test_data)))
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (Feats.shape[0], Feats.shape[-1]))
    return Feature