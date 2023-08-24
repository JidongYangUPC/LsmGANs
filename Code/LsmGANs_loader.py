import scipy.misc
from glob import glob

import numpy as np
import scipy.io as sio

class DataLoader_new():
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_batch_new(self, batch_size=1):

        path_A = sio.loadmat('datasets/Input/migrated_image.mat')
        path_A = path_A['migrated_image']
        path_A = path_A.reshape(path_A.shape[0], path_A.shape[1], path_A.shape[2], 1)

        path_B = sio.loadmat('datasets/Input/Source_illumination.mat')
        path_B = path_B['Source_illuminationn']
        path_B = path_B.reshape(path_B.shape[0], path_B.shape[1], path_B.shape[2], 1)

        path_C = sio.loadmat('datasets/Input/migration_velocity.mat')
        path_C = path_C['migration_velocity']
        path_C = path_C.reshape(path_C.shape[0], path_C.shape[1], path_C.shape[2], 1)


        path_D = sio.loadmat('datasets/Output/true_reflectivity.mat')
        path_D = path_D['true_reflectivity']
        path_D = path_D.reshape(path_D.shape[0], path_D.shape[1], path_D.shape[2], 1)


        indices_A = list(range(path_A.shape[0]))
        np.random.shuffle(indices_A)  # shuffle



        for i in range(0, len(indices_A), batch_size):
            batch_A=path_A[indices_A[i:i+batch_size]]
            batch_B=path_B[indices_A[i:i+batch_size]]
            batch_C=path_C[indices_A[i:i+batch_size]]
            batch_D = path_D[indices_A[i:i + batch_size]]
            yield batch_A, batch_B, batch_C, batch_D