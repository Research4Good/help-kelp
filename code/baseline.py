#
#
#
# https://www.kaggle.com/code/prambim/tf-autoencoders-landsat-cold-springs-fire
#

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

# Define the autoencoder model
input_size = [350,350,7]  # number of input features
encoding_size = 3  # size of the encoded representation

# Input layer
input_layer = Input(shape=(input_size,))

# Encoding layers
encoded = Dense(encoding_size * 16, activation='relu')(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_size * 8, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_size * 4, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_size * 2, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_size, activation='sigmoid')(encoded)  # Last layer of encoding with sigmoid

# Decoding layers
decoded = Dense(encoding_size * 2, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(encoding_size * 4, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(encoding_size * 8, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(encoding_size * 16, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(input_size, activation='sigmoid')(decoded)  # Last layer of decoding with sigmoid

# Autoencoder model
autoencoder = Model(input_layer, decoded)


# FLATTENING THE DATA
# Reshape the data
num_samples = stacked.shape[0] * stacked.shape[1]
num_features = stacked.shape[2]
stacked_2d = np.reshape(stacked, (num_samples, num_features))

print(stacked_2d.shape)  # Should print (43542, 7)

# COMPILING THE MODEL
# Importing necessary libraries
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from tensorflow.keras import backend as K

# Define R-Squared
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Compile the model
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=[r_squared])

# TRAINING AND DISPLAYING THE RESULT
# Importing necessary libraries

from skimage import exposure
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping

# Define a custom callback
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Predict the data
        predicted = self.model.predict(stacked_2d)

        # Reshape the predicted data to the original shape
        predicted_3d = np.reshape(predicted, stacked.shape)

        # Plot the original image
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        # Perform contrast stretching on the original image
        p10, p90 = np.percentile(stacked[:, :, [3, 2, 1]], (10, 90))
        img_rescale = exposure.rescale_intensity(stacked[:, :, [3, 2, 1]], in_range=(p10, p90))
        plt.imshow(img_rescale)
        plt.title('Original Image')

        # Plot the encoded image
        encoder = Model(input_layer, encoded)
        encoded_imgs = encoder.predict(stacked_2d)
        encoded_imgs_3d = np.reshape(encoded_imgs, (stacked.shape[0], stacked.shape[1], encoding_size))
        plt.subplot(1, 3, 2)
        plt.imshow(encoded_imgs_3d[:, :, [0, 1, 2]])
        plt.title('Encoded Image')

        # Plot the decoded image
        plt.subplot(1, 3, 3)
        # Perform contrast stretching on the decoded image
        p10, p90 = np.percentile(predicted_3d[:, :, [3, 2, 1]], (10, 90))
        img_rescale = exposure.rescale_intensity(predicted_3d[:, :, [3, 2, 1]], in_range=(p10, p90))
        plt.imshow(img_rescale)
        plt.title('Decoded Image')

        plt.show()

# Define the callback
callback = [EarlyStopping(monitor='loss', patience=3), CustomCallback()]

# Train the model with callback
history = autoencoder.fit(stacked_2d, stacked_2d, epochs=20, batch_size=32, callbacks=callback)
