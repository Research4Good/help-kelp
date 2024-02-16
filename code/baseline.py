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
