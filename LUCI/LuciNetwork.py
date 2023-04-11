"""
File containing the definitions required to train our Bayesian CNN
The functions are taken from https://keras.io/examples/keras_recipes/bayesian_neural_networks/#probabilistic-bayesian-neural-networks
"""
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

def create_MDN_model(input_length, loss):
    """
    This function constructs the MDN using the tensorflow Sequential API. The network has two convolutional
    layers, a max pooling layer, a flattening layer, a dropout layer, and two hidden layers until it finally
    outputs the mu and sigma values of the velocity and velocity dispersion parameters.

    The first convolutional layer has 4 filters of size 5. The second convolutional
    layers has 16 filters of size 3. The dropout is set to 20%. The first hidden layer
    has 128 nodes, and the second hidden layer has 256 nodes. Four parameters are outputted:
    mu_vel, mu_broad, sigma_vel, sigma_broad.

    Args:
        input_length: Length of input spectrum

    Return:
        Mixture Density Network with a single phase for two parameters
    """
    # Number of nodees in each hidden layer
    hidden_units = [128, 256]
    # Number of filters
    num_filters = [4, 16]
    # Length of filters
    filter_length = [5, 3]
    lr = 0.0007  # initial learning rate
    beta_1 = 0.9  # exponential decay rate  - 1st
    beta_2 = 0.999  # exponential decay rate  - 2nd
    optimizer_epsilon = 1e-08  # For the numerical stability
    # Define input which is a vector with 515 elements representing the spectra
    inputs = keras.Input(shape=(input_length,1))
    features = keras.layers.BatchNormalization()(inputs)
    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for filter_, length_ in zip(num_filters, filter_length):
        features = keras.layers.Conv1D(filters=filter_, kernel_size=length_, padding='same', activation='relu')(features)
    features = keras.layers.MaxPooling1D(pool_size=2)(features)
    features = keras.layers.Flatten()(features)
    features = keras.layers.Dropout(0.2)(features)
    for units in hidden_units:
        features = keras.layers.Dense(units, activation="relu")(features)
    distribution_params = keras.layers.Dense(units=4)(features)
    outputs = tfp.layers.IndependentNormal(2)(distribution_params)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    return model


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)
