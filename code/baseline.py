#
#
#
# https://www.kaggle.com/code/prambim/tf-autoencoders-landsat-cold-springs-fire
#

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

import rasterio
from glob import glob
from pathlib import Path



img_dir = 'kelp/train_features.tar_MLIC14m/train_satellite' 
lab_dir = 'kelp/train_labels.tar_l8u2RP0/train_kelp'
 
meta_df=pd.read_csv('kelp/metadata_fTq0l2T.csv')
meta_df.query('type == "kelp"')
meta_df=meta_df.sort_values('filename').query( 'in_train == True')
meta_df.head()


file = Path( lab_dir, meta_df['tile_id'].values[0] + '_kelp.tif' )
with rasterio.open(file) as src:    
    label=src.read(1)

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
