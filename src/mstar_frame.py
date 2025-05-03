import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.utils import utils
from utils.plot import plot

from typing import Tuple, Union
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Define MSTAR dataframe & associated utilities
class mstar_frame(utils, plot):

    # Initialize MSTAR dataframe parameters.
    def __init__(self, ctype: str='cart') -> None:

        # Inherit parent utilities class.
        super().__init__()

        # Initialize data processing parameters.
        self.ctype = ctype

    
    # Preprocess MSTAR dataset.
    def get_mstar_data(self, plot_flag: bool=False) -> Union[Tuple[np.array, np.array, np.array, np.array],
                                                             Tuple[np.array, np.array, np.array, np.array, np.array, np.array]]:

        # Generate lists of MSTAR dataset files.
        file_list_train = self.get_mstar_file_list(subset='train')  # Training dataset file list
        file_list_test = self.get_mstar_file_list(subset='test')    # Test dataset file list

        # Determine MSTAR image dimensions.
        header_0, _ = self.unpack_mstar_file(file_list_train[0])

        n_rg = header_0['NumberOfRows']                             # Number of row pixels
        n_az = header_0['NumberOfColumns']                          # Number of column pixels

        # Initialize output data and classification arrays.
        n_train = file_list_train.size
        n_test = file_list_test.size

        x_train_iq = np.zeros((n_train, n_rg, n_az, 1), dtype=np.complex64)
        x_test_iq = np.zeros((n_test, n_rg, n_az, 1), dtype=np.complex64)

        x_train_mp = np.zeros((n_train, n_rg, n_az, 1), dtype=np.complex64)
        x_test_mp = np.zeros((n_test, n_rg, n_az, 1), dtype=np.complex64)

        y_train = np.zeros((n_train,), dtype=object)
        y_test = np.zeros((n_test,), dtype=object)

        # Preprocess MSTAR training data.
        pbar_train = tqdm(range(n_train))
        pbar_train.set_description('Unpacking MSTAR training data')

        for i in pbar_train:

            header_train_i, data_train_iq_i, data_train_mp_i = self.unpack_mstar_file(file_list_train[i], ctype='all')

            x_train_iq[i, :, :, :] = np.reshape(data_train_iq_i, (n_rg, n_az, 1))
            x_train_mp[i, :, :, :] = np.reshape(data_train_mp_i, (n_rg, n_az, 1))

            y_train[i] = header_train_i['TargetType']
        
        pbar_train.close()

        # Preprocess MSTAR test data.
        pbar_test = tqdm(range(n_test))
        pbar_test.set_description('Unpacking MSTAR test data')

        for i in pbar_test:

            header_test_i, data_test_iq_i, data_test_mp_i = self.unpack_mstar_file(file_list_test[i], ctype='all')

            x_test_iq[i, :, :, :] = np.reshape(data_test_iq_i, (n_rg, n_az, 1))
            x_test_mp[i, :, :, :] = np.reshape(data_test_mp_i, (n_rg, n_az, 1))

            y_test[i] = header_test_i['TargetType']
        
        pbar_test.close()

        # Plot one example from each target classification.
        if plot_flag:

            self.plot_heatmap(x_train_iq[10])      # BMP2
            self.plot_heatmap(x_train_iq[350])     # BTR70
            self.plot_heatmap(x_train_iq[750])     # T72

        # Recombine to shuffle and split training and test data.
        x_full_iq = np.concatenate((x_train_iq, x_test_iq), axis=0)
        x_full_mp = np.concatenate((x_train_mp, x_test_mp), axis=0)

        y_full = np.concatenate((y_train, y_test), axis=0)

        x_train_iq, x_test_iq, y_train, y_test = train_test_split(x_full_iq, y_full, test_size=0.25, random_state=42)
        x_train_mp, x_test_mp, _, _ = train_test_split(x_full_mp, y_full, test_size=0.25, random_state=42)

        # Format output complex image data.
        x_train_iq, x_test_iq = self.to_tensor(x_train_iq, x_test_iq)
        x_train_mp, x_test_mp = self.to_tensor(x_train_mp, x_test_mp)

        if ((self.ctype.lower() == 'cart') or (self.ctype.lower() == 'cartesian') or
            (self.ctype.lower() == 'rect') or (self.ctype.lower() == 'rectangular')):

            return (x_train_iq, y_train, x_test_iq, y_test)
        
        elif ((self.ctype.lower() == 'pol') or (self.ctype.lower() == 'polar')):

            return (x_train_mp, y_train, x_test_mp, y_test)
        
        else:

            return (x_train_iq, x_train_mp, y_train, x_test_iq, x_test_mp, y_test)

