import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable debugging and warning logs

from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.utils import utils
from utils.plot import plot


#%% Define model class & associated utilities
class dnn_model(utils, plot):

    # Initialize model parameters.
    def __init__(self, \
                 train_df: tf.Tensor, train_id: np.array, \
                 test_df: tf.Tensor, test_id: np.array, \
                 filter_size: int=24, learning_rate: float=0.0001, \
                 ctype: str='iq', init_flag: bool=True) -> None:

        # Inherit parent utilities class.
        super().__init__()

        # Initialize MSTAR dataframe parameters.
        self.train_df = tf.convert_to_tensor(train_df, dtype=tf.float32)                                # Training data
        self.train_id = train_id                                                                        # Training data classification label

        self.test_df = tf.convert_to_tensor(test_df, dtype=tf.float32)                                  # Test data
        self.test_id = test_id                                                                          # Test data classification label

        self.train_id_oh = tf.convert_to_tensor(self.to_one_hot(self.train_id), dtype=tf.float32)       # One-hot encoded training classification label
        self.test_id_oh = tf.convert_to_tensor(self.to_one_hot(self.test_id), dtype=tf.float32)         # One-hot encoded test classification label 

        self.train_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.train_id_oh)[1][:], \
                                                               (self.train_id.size, 1)))                # Sparse training classification label
        self.test_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.test_id_oh)[1][:], \
                                                              (self.test_id.size, 1)))                  # Sparse test classification label

        self.n_cov = self.train_df.shape[1]                                                              # Number of covariates

        self.list_id = np.unique(self.train_id)                                                         # List of unique classifications
        self.n_id = self.list_id.size                                                                   # Number of unique classifications

        # Initialize model parameters.
        self.init_flag = init_flag                                                                      # Initialize new CNN model

        self.filter_size = filter_size                                                                  # Convolutional layer filter size
        self.learning_rate = learning_rate                                                              # Initial learning rate for weight updates
        
        self.ctype = ctype                                                                              # Complex component type (I, Q, magnitude, phase)

        # Initialize CNN model.
        self.model = self.get_model(filter_size=self.filter_size, \
                                    learning_rate=self.learning_rate, \
                                    verbose=False)                                                      # Real-valued neural network (CNN) model                


    # Generate CNN model.
    def get_model(self, filter_size: int=24, learning_rate: float=0.0001, verbose: bool=True) -> Model:

        # Create new CNN model.  
        if self.init_flag:

            # Input weighted model predictions of shape [n_cov, 1].
            IN = Input(shape=(self.n_cov, 1))

            # Hidden dense layers.
            D1 = Dense(filter_size, activation='sigmoid')(IN)
            D2 = Dense(filter_size, activation='sigmoid')(D1)

            # Output one-hot encoded classification of shape [n_id, 1].
            F = Flatten()(D2)
            OUT = Dense(self.n_id, activation='softmax')(F)

            dnn = Model(inputs=IN, outputs=OUT, name=str('dnn_' + self.ctype.lower()))

            # Compile model for training.
            dnn.compile(loss='categorical_crossentropy', \
                        optimizer=Adam(learning_rate=learning_rate), \
                        metrics=['accuracy'])

        # Load existing CVNN model.
        else:

            dnn = load_model(str('dnn_' + self.ctype.lower() + '.keras'))

        # Print model summary, if desired.
        if verbose:

            dnn.summary()

        return dnn
    

    # Train CNN model on MSTAR dataset.
    def train(self, \
              n_epoch: int=25, batch_size: int=64, patience: int=3, \
              verbose: bool=True, plot_flag: bool=True) -> Tuple[float, float, float, float]:
        
        # Define callbacks to optimize model training.
        es = EarlyStopping(monitor='val_loss', \
                           verbose=verbose, \
                           patience=patience, \
                           min_delta=0.001)                                 # Stop training at right time
        mc = ModelCheckpoint(str('dnn_' + self.ctype.lower() + '.keras'), \
                             monitor='val_accuracy', \
                             verbose=verbose, \
                             save_best_only='True', \
                             mode='max')                                    # Save best model after each training epoch
        red_lr = ReduceLROnPlateau(monitor='val_loss', \
                                   factor=0.1,
                                   patience=int(patience/2),
                                   min_lr=0.000001)                         # Reduce learning rate once learning stagnates
        
        # Perform model training.
        if ((self.ctype.lower() == 'iq') or (self.ctype.lower() == 'mp')):
            
            fit_info = self.model.fit(x=self.train_df, y=self.train_id_oh, \
                                      epochs=n_epoch, batch_size=batch_size, \
                                      callbacks=[es, mc, red_lr], \
                                      validation_split=0.1)
        
        else:

            raise ValueError('Invalid phase component type for real-valued model.')
            
        # Calculate model training diagnostics.
        train_loss = np.array(fit_info.history['loss'], dtype=float)
        val_loss = np.array(fit_info.history['val_loss'], dtype=float)

        train_acc = np.array(fit_info.history['accuracy'], dtype=float)
        val_acc = np.array(fit_info.history['val_accuracy'], dtype=float)

        # Generate model diagnostics plots, if desired.
        if plot_flag:

            # Plot training/validation loss over each training epoch.
            plt.plot(np.arange(train_loss.size), train_loss, color='blue', label=('Training Loss (' + str(self.ctype) + ')'))
            plt.plot(np.arange(val_loss.size), val_loss, color='red', linestyle='dashed', label=('Validation Loss (' + str(self.ctype) + ')'))
            
            plt.legend()

            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.show()

            # Plot training/validation accuracy over each training epoch.
            plt.plot(np.arange(train_acc.size), train_acc, color='blue', label=('Training Accuracy (' + str(self.ctype) + ')'))
            plt.plot(np.arange(val_acc.size), val_acc, color='red', linestyle='dashed', label=('Validation Accuracy (' + str(self.ctype) + ')'))
            
            plt.legend()

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

            plt.show()
        
        # Calculate training metrics in final model.
        f = np.argmax(val_acc)

        final_train_loss = train_loss[f]
        final_val_loss = val_loss[f]

        final_train_acc = train_acc[f]
        final_val_acc = val_acc[f]

        return (final_train_loss, final_val_loss, final_train_acc, final_val_acc)
    

    # Generate CNN model predictions for multiple classification.
    def predict(self, x: np.array):

        return np.argmax(self.model.predict(x), axis=1)


    # Evaluate CNN model performance.
    def evaluate(self, x: tf.Tensor, y: tf.Tensor, plot_flag: bool=False) -> Tuple[float, float, float, float]:

        # Generate predictions on the test set.
        yhat = self.predict(x)

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

