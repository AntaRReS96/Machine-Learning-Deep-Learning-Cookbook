import tensorflow as tf


def make_model(input_shape, hidden_size, output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(output_size, activation="linear")  # linear dla Q-values
    ])

    model.build(input_shape=(None, 16))
    return model
