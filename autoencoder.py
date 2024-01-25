import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Encoder
x = tf.keras.layers.Input(shape=(784), name="encoder_input")
encoder_dense_layer1 = tf.keras.layers.Dense(units=1024, name="encoder_dense_1")(x)
encoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)
encoder_dense_layer2 = tf.keras.layers.Dense(units=512, name="encoder_dense_2")(encoder_activ_layer1)
encoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_2")(encoder_dense_layer2)
encoder_dense_layer3 = tf.keras.layers.Dense(units=256, name="encoder_dense_3")(encoder_activ_layer2)
encoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_3")(encoder_dense_layer3)
encoder_output = tf.keras.layers.LeakyReLU(name="encoder_output")(encoder_activ_layer3)

encoder = tf.keras.models.Model(x, encoder_output, name="encoder_model")
encoder.summary()

# Decoder
decoder_input = tf.keras.layers.Input(shape=(256), name="decoder_input")
decoder_dense_layer1 = tf.keras.layers.Dense(units=512, name="decoder_dense_1")(decoder_input)
decoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)
decoder_dense_layer2 = tf.keras.layers.Dense(units=1024, name="decoder_dense_2")(decoder_activ_layer1)
decoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_dense_layer2)
decoder_dense_layer3 = tf.keras.layers.Dense(units=784, name="decoder_dense_3")(decoder_activ_layer2)
decoder_output = tf.keras.layers.LeakyReLU(name="decoder_output")(decoder_dense_layer3)

decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
decoder.summary()

# Autoencoder
ae_input = tf.keras.layers.Input(shape=(784), name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)

ae = tf.keras.models.Model(ae_input, ae_decoder_output, name="AE")
ae.summary()

# RMSE
def rmse(y_true, y_predict):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_predict))

# AE Compilation
ae.compile(loss=rmse, optimizer=tf.keras.optimizers.Adam(lr=0.0005))

# Preparing MNIST Dataset
(x_train_orig, y_train), (x_test_orig, y_test) = tf.keras.datasets.mnist.load_data()
x_train_orig = x_train_orig.astype("float32") / 255.0
x_test_orig = x_test_orig.astype("float32") / 255.0

x_train = np.reshape(x_train_orig, newshape=(x_train_orig.shape[0], np.prod(x_train_orig.shape[1:])))
x_test = np.reshape(x_test_orig, newshape=(x_test_orig.shape[0], np.prod(x_test_orig.shape[1:])))

# Training AE for more epochs
ae.fit(x_train, x_train, epochs=150, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Save the Autoencoder model
ae.save("autoencoder_model_updated.h5")

# Visualizations
encoded_images = encoder.predict(x_train)
decoded_images = decoder.predict(encoded_images)
decoded_images_orig = np.reshape(decoded_images, newshape=(decoded_images.shape[0], 28, 28))

num_images_to_show = 5
for im_ind in range(num_images_to_show):
    plot_ind = im_ind * 2 + 1
    rand_ind = np.random.randint(low=0, high=x_train.shape[0])
    plt.subplot(num_images_to_show, 2, plot_ind)
    plt.imshow(x_train_orig[rand_ind, :, :], cmap="gray")
    plt.subplot(num_images_to_show, 2, plot_ind + 1)
    plt.imshow(decoded_images_orig[rand_ind, :, :], cmap="gray")

plt.figure()
plt.scatter(encoded_images[:, 0], encoded_images[:, 1], c=y_train)
plt.colorbar()

# Show the visualizations
plt.show()

# Print the final RMSE on the test data
final_loss = ae.evaluate(x_test, x_test)
print("Final RMSE on Test Data:", final_loss)

