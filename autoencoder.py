import imp
import keras
import tensorflow as tf

class AutoEncoder:

    def __init__(self, nb_channels_in, nb_channels_out, unet_filters, last_activation, image_size=[None,None], latent_dim = 128):
        inputs = keras.Input([*image_size, nb_channels_in])
        
        self.input_shape = image_size
        self.latent_dim = latent_dim

        def conv_block(filters, strides, last_activation=None):
            if last_activation == None:
                conv_block = keras.Sequential([keras.layers.Conv2D(filters=filters, kernel_size=3, 
                                                                strides=1, padding="same", 
                                                                kernel_initializer = "he_normal"),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.LeakyReLU(), 
                                            keras.layers.Conv2D(filters=filters, kernel_size=3, 
                                                                strides=strides, padding="same", 
                                                                kernel_initializer = "he_normal"), 
                                            keras.layers.BatchNormalization(), 
                                            keras.layers.LeakyReLU()])
            else:
                conv_block = keras.Sequential([keras.layers.Conv2D(filters=filters, kernel_size=3, 
                                                                strides=strides, padding="same", 
                                                                kernel_initializer = "he_normal"),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Activation(last_activation)])
            return conv_block

        def up_sampler(filters):
            up_sampler = keras.Sequential([keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, 
                                                                        strides=2, padding="valid", 
                                                                        kernel_initializer = "he_normal"), 
                                        keras.layers.BatchNormalization(), 
                                        keras.layers.LeakyReLU()])
            return up_sampler

        for i in range(len(unet_filters)):
            if i == 0:
                x = conv_block(unet_filters[i], 1)(inputs)

            else:
                x = conv_block(unet_filters[i], 2)(x)
        
        x = keras.layers.Flatten()(x)
        latent = keras.layers.Dense(latent_dim, name="latent")(x)

        self.encoder = keras.Model(inputs,latent)

        x = keras.layers.Dense(288)(latent)
        x = keras.layers.Reshape((6, 6, 8))(x)

        for i in reversed(range(len(unet_filters)-1)):
            x = up_sampler(unet_filters[i])(x)
            x = conv_block(unet_filters[i], 1)(x)
        
        outputs = conv_block(nb_channels_out, 1, last_activation)(x)

        self.decoder = keras.Model(latent,outputs)

        self.autoencoder = keras.Model(inputs, outputs)
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='mse')
        
    def fit(self, data_x, data_y, epochs, batch_size, validation_split=0.2, shuffle = True, callbacks = []):
        return self.autoencoder.fit(x=data_x, 
                            y=data_y, 
                            validation_split=validation_split,
                            shuffle=shuffle, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            callbacks=callbacks)
        