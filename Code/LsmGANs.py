from __future__ import print_function, division
import scipy
import scipy.io as sio
import tensorflow as tf
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers import Input, Layer, Conv2DTranspose, add, Reshape, Flatten, Dropout, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
#from keras_contribcontrib.layers.normalization import InputSpec
import datetime
import matplotlib.pyplot as plt
import sys
from LsmGANs_loader import DataLoader_new
import numpy as np
import os
import json
from keras.layers import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.normalization = InstanceNormalization
        self.img_shape= (self.img_rows, self.img_cols, self.channels)
        # self.normalization = InstanceNormalization
        # Configure data loader
        self.dataset_name = 'migratedorR'
        self.data_loader = DataLoader_new(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))



        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**3)
        self.disc_patch = (patch,patch,1)


        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        optimizer = Adam(0.0002, 0.5)
        self.g_AB = self.build_generator(name='g_AB_model')
        self.g_AB.summary()
        self.d_A = self.build_discriminator(name='d_A_model')
        self.d_A.summary()
        # Build and compile the discriminators
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])


        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators

        # Input images from both domains
        img=Input(shape=self.img_shape)
        label1=Input(shape=self.img_shape)
        label2 = Input(shape=self.img_shape)
        fake= self.g_AB([img,label1,label2])

        # For the combined model we will only train the generators
        self.d_A.trainable = False

        # Discriminators determines validity of translated images

        valid = self.d_A([fake,label1,label2])

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img,label1,label2],
                              outputs=valid)
        self.combined.compile(loss='mse',optimizer=optimizer)
#======================================================================================================
# Architecture functions
    def ck(self, x, k, use_normalization, stride,m):
        x = Conv2D(filters=k, kernel_size=m, strides=stride, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # x = LeakyReLU(alpha=0.2)(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        # x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])


        return x

    def uk(self, x, k):
        # x = ReflectionPadding2D((1, 1))(x)
        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        # x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # x = LeakyReLU(alpha=0.2)(x)

        return x

    # ===============================================================================

    def build_generator(self,name=None):

        input_img_lsrtm = Input(shape=self.img_shape)
        label1 = Input(shape=self.img_shape)
        label2 = Input(shape=self.img_shape)

        input_img_multiply = Multiply()([input_img_lsrtm, label1])
        input_img_multiply_Concatenate = Concatenate(axis=-1)([input_img_multiply, label2])
        # x0 = ReflectionPadding2D((3, 3))(input_img_multiply)  # (256,256,1)
        # Layer 1
        x1 = self.c7Ak(input_img_multiply_Concatenate, 32)#size(256,256,32)
        # Layer 2
        x2 = self.dk(x1, 64)#size(128,128,64)
        # Layer 3
        x3 = self.dk(x2, 128)#size(64,64,128)
        # Layer 4
        # x4 = self.dk(x3, 256)  # size(32,32,256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):#size(63,63,128)
            x3 = self.Rk(x3)

        # Layer 13
        x5 = self.uk(x3, 64)
        # x6=Concatenate()([x5, x3])
        # Layer 14
        x6 = self.uk(x5, 32)
        # x8 = Concatenate()([x7,x2])
        # x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1,padding='same')(x6)
        output_img = Activation('tanh')(x)  # They say they use Relu but really they do not

        return Model(inputs=[input_img_lsrtm, label1,label2], outputs=output_img, name=name)


    def build_discriminator(self,name=None):

        # Specify input
        generated_img_lsrtm = Input(shape=self.img_shape)
        label1 = Input(shape=self.img_shape)
        label2 = Input(shape=self.img_shape)
        input_img_multiply = Multiply()([generated_img_lsrtm, label1])#size(256,256)
        input_img_multiply_Concatenate = Concatenate(axis=-1)([input_img_multiply, label2])

        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img_multiply_Concatenate, 64, False, 1,3)#size(128,128,64)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Layer 2
        x = self.ck(x, 128, True, 1,3)#size(64,64,128)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Layer 3
        x = self.ck(x, 256, True, 1,3)#size(32,32,256)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Layer 4
        x = self.ck(x, 512, True, 1,4)#size(32,32,512)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)#size(32,32,1)
        # validity = Activation('sigmoid')(x)

        return Model(inputs=[generated_img_lsrtm, label1, label2], outputs=validity, name=name)
if __name__ == '__main__':
    gan = CGAN()

#     def train(self, epochs, batch_size=1, sample_interval=100):
#
#         # Adversarial loss ground truths
#         valid = np.ones((batch_size,)+self.disc_patch)
#         fake = np.zeros((batch_size,)+self.disc_patch)
#
#         for epoch in range(epochs):
#             for batch_i, (migrated,label1,label2,reflectivity) in enumerate(self.data_loader.load_batch_new(batch_size)):
#
#                 # ----------------------
#                 #  Train Discriminators
#                 # ----------------------
#                 # Translate images to opposite domain
#                 gen_reflectivity = self.g_AB.predict([migrated, label1, label2])
#
#                 # Train the discriminators (original images = real / translated = Fake)
#                 dA_loss_real = self.d_A.train_on_batch([reflectivity, label1, label2],valid)
#                 dA_loss_fake = self.d_A.train_on_batch([gen_reflectivity, label1, label2],fake)
#
#                 d_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
#
#
#                 # ------------------
#                 #  Train Generators
#                 g_loss = self.combined.train_on_batch([migrated, label1, label2],valid)
#
#                 # Plot the progress
#                 print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
#
#
#                 # If at save interval => save generated image samples
#                 if batch_i % sample_interval== 0:
#                     self.sample_images(epoch, batch_i)
#             #### Store modles
#
#             self.saveModel(self.g_AB,epoch)
#
#
#     def saveModel(self,model,epoch):
#         directory=os.path.join('saved_LsmGANs')
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#
#
#         model_path_w = 'saved_LsmGANs/{}_weights_epoch_{}.hdf5'.format(model.name, epoch)
#         model.save_weights(model_path_w)
#
#
#         model_path_m = 'saved_LsmGANs/{}_weights_epoch_{}.json'.format(model.name, epoch)
#         json_string = model.to_json()
#         with open(model_path_m, 'w') as outfile:
#             json.dump(json_string, outfile)
#
#         print('The {} epoch of {} model has been saved'.format(epoch,model.name))
#
#
#     def sample_images(self, epoch, batch_i):
#         r, c = 1, 2
#
#         Input_test = sio.loadmat('datasets/Input/migrated_image.mat')
#         Input_test = Input_test['migrated_image']
#         test_A = Input_test.reshape((Input_test.shape[0], Input_test.shape[1], Input_test.shape[2], 1))
#
#         Input_test = sio.loadmat('datasets/Input/Source_illumination.mat')
#         Input_test = Input_test['Source_illumination']
#         label1 = Input_test.reshape((Input_test.shape[0], Input_test.shape[1], Input_test.shape[2], 1))
#
#         Input_test = sio.loadmat('datasets/Input/migration_velocity.mat')
#         Input_test = Input_test['migration_velocity']
#         label2 = Input_test.reshape((Input_test.shape[0], Input_test.shape[1], Input_test.shape[2], 1))
#
#         Output_test = sio.loadmat('datasets/Output/true_reflectivity.mat')
#         Output_test = Output_test['true_reflectivity']
#         test_B_1 = Output_test.reshape((Output_test.shape[0], Output_test.shape[1], Output_test.shape[2], 1))
#
#         # Translate images to the other domain
#         fake_mar_B = self.g_AB.predict([test_A,label1,label2])
#
#         A = list(range(3))
#         i=np.random.choice(A,1)
#         test_A_mar=test_A[i,:,:,:].reshape(1,256,256)
#         fake_B_mar=fake_mar_B[i, :, :,:].reshape(1,256, 256)
#         reflectivity= test_B_1[i,:,:].reshape(1,256,256)
#
#         gen_imgs_mar = np.concatenate([test_A_mar, fake_B_mar, reflectivity])
#
#         titles = ['migrated', 'Predicted', 'Reflectivity']
#         fig, axs = plt.subplots(r, c)
#         cnt = 0
#         for i in range(c):
#             axs[i].imshow(gen_imgs_mar[cnt],cmap=plt.cm.gray,vmin=-0.3,vmax=0.3)
#             axs[i].set_title(titles[i])
#             axs[i].axis('off')
#             cnt += 1
#
#         fig.savefig("images/%d_%d.png" % (epoch, batch_i),bbox_inches='tight')
#         plt.close()
#
# if __name__ == '__main__':
#     gan = CGAN()
#     gan.train(epochs=300, batch_size=1, sample_interval=100)


    Input_validation_Marmousi = sio.loadmat('migrated_image.mat')
    Input_validation_Marmousi = Input_validation_Marmousi['migrated_image']
    Input_validation_Marmousi = Input_validation_Marmousi.reshape((Input_validation_Marmousi.shape[0], Input_validation_Marmousi.shape[1], Input_validation_Marmousi.shape[2], 1))


    Hessian_patch = sio.loadmat('Source_illumination.mat')
    Hessian_patch = Hessian_patch['Source_illumination']
    Hessian_label = Hessian_patch.reshape((Hessian_patch.shape[0], Hessian_patch.shape[1], Hessian_patch.shape[2], 1))

    Vpsm_patch = sio.loadmat('migration_velocity.mat')
    Vpsm_patch = Vpsm_patch['migration_velocity']
    Vpsm_label = Vpsm_patch.reshape((Vpsm_patch.shape[0], Vpsm_patch.shape[1], Vpsm_patch.shape[2], 1))


    g_AB_model=gan.build_generator(name='g_AB_model')
    g_AB_model.load_weights('saved_LsmGANs/{}_weights_epoch_{}.hdf5'.format(g_AB_model.name, 299))
    print("Loaded model from disk")


    #  Translate back to original domain
    CNN_reflectivity= g_AB_model.predict([Input_validation_Marmousi,Hessian_label,Vpsm_label])
    dataNew = 'Result/CNN_reflectivity.mat'
    sio.savemat(dataNew, {'CNN_reflectivity': CNN_reflectivity})




