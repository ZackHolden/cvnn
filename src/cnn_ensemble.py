import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable debugging and warning logs
import csv
from typing import Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import differential_evolution

from src.cnn_model import cnn_model
from src.dnn_model import dnn_model


#%% Define ensemble class & associated utilities
class cnn_ensemble(cnn_model):

    # Initialize model parameters.
    def __init__(self, \
                 train_df: tf.Tensor, train_id: np.array, \
                 test_df: tf.Tensor, test_id: np.array, \
                 filter_size: int=24, kernel_size: int=5, \
                 dropout_rate: float=0.1, \
                 pol_flag: bool=False, init_flag: bool=True, train_flag: bool=False) -> None:

        # Inherit parent utilities class.
        super().__init__(train_df=train_df, train_id=train_id, \
                         test_df=test_df, test_id=test_id)

        # Initialize MSTAR dataframe parameters.
        self.train_df = train_df                                                                        # Complex training data
        self.train_id = train_id                                                                        # Training data classification label

        self.test_df = test_df                                                                          # Complex test data
        self.test_id = test_id                                                                          # Test data classification label

        self.n_rg = self.train_df.shape[1]                                                              # Number of range pixels
        self.n_az = self.train_df.shape[2]                                                              # Number of azimuth pixels

        self.list_id = np.unique(self.train_id)                                                         # List of unique classifications
        self.n_id = self.list_id.size                                                                   # Number of unique classifications

        self.train_df_i = tf.convert_to_tensor(tf.math.real(self.train_df), dtype=tf.float32)           # Real component (I) of training data
        self.train_df_q = tf.convert_to_tensor(tf.math.imag(self.train_df), dtype=tf.float32)           # Imaginary component (Q) of training data

        self.test_df_i = tf.convert_to_tensor(tf.math.real(self.test_df), dtype=tf.float32)             # Real component (I) of test data
        self.test_df_q = tf.convert_to_tensor(tf.math.imag(self.test_df), dtype=tf.float32)             # Imaginary component (Q) of test data                       

        self.train_id_oh = tf.convert_to_tensor(self.to_one_hot(self.train_id), dtype=tf.float32)       # One-hot encoded training classification label
        self.test_id_oh = tf.convert_to_tensor(self.to_one_hot(self.test_id), dtype=tf.float32)         # One-hot encoded test classification label 

        self.train_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.train_id_oh)[1][:], \
                                                               (self.train_id.size, 1)))                # Sparse training classification label
        self.test_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.test_id_oh)[1][:], \
                                                              (self.test_id.size, 1)))                  # Sparse test classification label

        # Initialize model parameters.
        self.init_flag = init_flag                                                                      # Initialize new CNN ensemble weights
        self.train_flag = train_flag                                                                    # Retrain output DNN model

        self.filter_size = filter_size                                                                  # Convolutional layer filter size
        self.kernel_size = kernel_size                                                                  # Convolutional layer kernel size
        self.dropout_rate = dropout_rate                                                                # Fraction of input units to drop

        if pol_flag:

            self.ctype = 'mp'                                                                           # Polar complex format

        else:

            self.ctype = 'iq'                                                                           # Cartesian complex format 

        # Initialize CNN ensemble.
        self.model_i = cnn_model(train_df=self.train_df, train_id=self.train_id, \
                                 test_df=self.test_df, test_id=self.test_id, \
                                 filter_size=self.filter_size, \
                                 kernel_size=self.kernel_size, \
                                 dropout_rate=self.dropout_rate, \
                                 ctype=self.ctype[0], \
                                 init_flag=False)                                                       # Real component neural network (CNN) model
                        
        self.model_q = cnn_model(train_df=self.train_df, train_id=self.train_id, \
                                 test_df=self.test_df, test_id=self.test_id, \
                                 filter_size=self.filter_size, \
                                 kernel_size=self.kernel_size, \
                                 dropout_rate=self.dropout_rate, \
                                 ctype=self.ctype[1], \
                                 init_flag=False)                                                       # Imaginary component neural network (CNN) model

        self.ensemble = [self.model_i, self.model_q]                                                    # I/Q CNN ensemble

        # Initialize average and weighted-average model weights.
        self.eq_wts = np.array((0.5, 0.5))                                                              # Average model weights

        if self.init_flag:

            self.opt_wts = self.get_opt_wts(self.test_df, self.test_id_oh)                              # Weighted average model weights
        
        else:

            # self.opt_wts = np.array([0.48449787, 0.51550213])                                           # Weighted average model weights
            self.opt_wts = self.unpack_opt_wts()

        # Generate weighted logits for output dense neural network.
        xhat_train = self.predict(self.train_df, self.opt_wts, argmax_flag=False)
        y_train = self.train_id

        xhat_test = self.predict(self.test_df, self.opt_wts, argmax_flag=False)
        y_test = self.test_id

        self.model_out = dnn_model(train_df=xhat_train, train_id=y_train, \
                                   test_df=xhat_test, test_id=y_test, \
                                   filter_size=256, learning_rate=0.01, \
                                   ctype=self.ctype, init_flag=self.train_flag)
        
        if self.train_flag:

            self.model_out.train()


    # Train output dense neural network in ensemble.
    def train(self, \
              n_epoch: int=25, batch_size: int=64, patience: int=3, \
              ctype: str='iq', verbose: bool=True) -> None:
        
        # Define callbacks to optimize model training.
        es = EarlyStopping(monitor='val_loss', \
                           verbose=verbose, \
                           patience=patience, \
                           min_delta=0.001)                                 # Stop training at right time
        mc = ModelCheckpoint(str('dnn_' + ctype.lower() + '.keras'), \
                             monitor='val_accuracy', \
                             verbose=verbose, \
                             save_best_only='True', \
                             mode='max')                                    # Save best model after each training epoch
        red_lr = ReduceLROnPlateau(monitor='val_loss', \
                                   factor=0.1,
                                   patience=int(patience/2),
                                   min_lr=0.000001)                         # Reduce learning rate once learning stagnates
        
        # Perform model training.
        self.model_out.fit(x=self.model_out.train_df, y=self.model_out.train_id, \
                           epochs=n_epoch, batch_size=batch_size, \
                           callbacks=[es, mc, red_lr], \
                           validation_split=0.1)
        

    # Generate CNN ensemble predictions for multiple classification.
    def predict(self, x: tf.Tensor, w: np.array=None, argmax_flag: bool=True) -> Tuple[float, float, float, float]:
        
        if w is None:

            w = self.opt_wts

        x_i = tf.convert_to_tensor(tf.math.real(x), dtype=tf.float32)
        x_q = tf.convert_to_tensor(tf.math.imag(x), dtype=tf.float32)
        
        yhat = np.array((self.model_i.model.predict(x_i), self.model_q.model.predict(x_q)))     # Ensemble fitted values
        prob_sum = np.tensordot(yhat, w, axes=((0), (0)))                                       # Weighted sum across ensemble members

        if argmax_flag:

            return np.argmax(prob_sum, axis=1)
        
        else:

            return prob_sum
        
    
    # Generate stacked ensemble predictions for multiple classification.
    def stack_predict(self, x: tf.Tensor, w: np.array=None):

        xhat = self.predict(x, w=w, argmax_flag=False)
        yhat = self.model_out.predict(xhat)

        return yhat
    

    # Evaluate CNN ensemble predictions.
    def evaluate(self, x: tf.Tensor, y: tf.Tensor, w: np.array=None, stack_flag: bool=True, plot_flag: bool=False) -> float:

        # Use pre-optimized weights if argument is None.
        if w is None:

            w = self.opt_wts
        
        # Generate predictions on the test set.
        if stack_flag:

            yhat = self.stack_predict(x, w)

        else:

            yhat = self.predict(x, w)

        # Convert ground truth classification to sparse categorical representation.
        y = np.argmax(y, axis=1)

        # Calculate accuracy, precision, recall, and F1 scores with global average.
        accuracy = accuracy_score(y, yhat)

        precision = precision_score(y, yhat, average='macro')
        recall = recall_score(y, yhat, average='macro')
        f1 = f1_score(y, yhat, average='macro')
        
        # Generate confusion matrix, if desired.
        if plot_flag:
            
            self.plot_confusion_matrix(y, yhat, np.array(['BMP2', 'BTR70', 'T72']))
        
        return (accuracy, precision, recall, f1)


    # Score function for CNN ensemble weight optimization with differential evolution.
    def score_fn(self, x: tf.Tensor, y: tf.Tensor, w: np.array) -> float:

        # Generate predictions on the test set.
        yhat = self.predict(x, w)

        # Convert ground truth classification to sparse categorical representation.
        y = np.argmax(y, axis=1)

        return(accuracy_score(y, yhat))
    

    # Loss function for CNN ensemble weight optimization with differential evolution.
    def loss_fn(self, w: np.array, x: tf.Tensor, y: tf.Tensor) -> float:

        wn = self.norm(w)                   # Normalized ensemble weights
        score = self.score_fn(x, y, w=wn)   # Ensemble error rate

        return (1.0 - score)
    

    # Perform CNN ensemble weight optimization.
    def get_opt_wts(self, \
                    x: tf.Tensor, y: tf.Tensor, \
                    iter_max: int=10, epsilon: float=0.01, \
                    save_flag: bool=True) -> np.array:

        # Define optimization parameters.
        bound = [(0.0, 1.0), (0.0, 1.0)]
        search_arg = (x, y)

        # Perform global optimization of ensemble weights.
        opt_wts = differential_evolution(self.loss_fn, bound, \
                                         args=search_arg, maxiter=iter_max, \
                                         tol=epsilon)
        opt_wts_norm = np.array(self.norm(opt_wts['x']))
        
        # Save optimized weights to CSV file, if desired.
        if save_flag:

            fname = 'opt_wts_' + self.ctype.lower() + '.csv'

            try:

                with open(fname, 'w', newline='', encoding='utf-8') as fw:

                    writer = csv.writer(fw)
                    writer.writerow(opt_wts_norm.tolist())

            except Exception:
        
                print('Invalid file to save optimized weights.')
        
        return opt_wts_norm
    

    # Unpack CNN ensemble pre-optimized weights.
    def unpack_opt_wts(self) -> np.array:

        # Initialize file parameters.
        fname = 'opt_wts_' + self.ctype.lower() + '.csv'

        # Unpack pre-optimized weights.
        try:

            with open(fname, 'r', encoding='utf-8') as fw:

                reader = csv.reader(fw)

                for row in reader:

                    opt_wts = np.array([float(x.strip().replace('\ufeff', '')) for x in row])

                    break   # Single row of data
        
        except Exception:
    
            raise ValueError('Error occurred while reading optimized weights.')

        return opt_wts

