import keras as kr
import numpy as np

from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model, Input, Sequential, load_model
import keras.activations as activations


class CNN:
    def __init__(self, RNNconfig):
        self.config = RNNconfig
        self.input_shape = (self.config['num_steps'], self.config['sensors'])
        self.input = Input(self.input_shape)


    def build_model(self, inception=True, res=True, strided=False, maxpool=True, avgpool=False, batchnorm=True):
        self.i = 0
        pad = 'same'
        padp = 'same'

        c_act = self.config['c_act']
        r_act = self.config['r_act']
        rk_act = self.config['rk_act']

        r = kr.regularizers.l2(self.config['reg'])

        c = self.input
        stride_size = self.config['strides'] if strided else 1

        if inception:
            c0 = layers.Conv1D(self.config['filters'], kernel_size=4, strides=stride_size, padding=pad,
                               activation=c_act)(c)
            c1 = layers.Conv1D(self.config['filters'], kernel_size=8, strides=stride_size, padding=pad,
                               activation=c_act)(c)
            c2 = layers.Conv1D(self.config['filters'], kernel_size=32, strides=stride_size, padding=pad,
                               activation=c_act)(c)

            c = layers.concatenate([c0, c1, c2])

            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)



            c0 = layers.Conv1D(self.config['filters'], kernel_size=4, strides=stride_size, padding=pad,
                               activation=c_act)(c)
            c1 = layers.Conv1D(self.config['filters'], kernel_size=8, strides=stride_size, padding=pad,
                               activation=c_act)(c)
            c2 = layers.Conv1D(self.config['filters'], kernel_size=32, strides=stride_size, padding=pad,
                               activation=c_act)(c)

            c = layers.concatenate([c0, c1, c2])
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

            c0 = layers.Conv1D(self.config['filters'], kernel_size=4, strides=stride_size, padding=pad,
                               activation=c_act)(c)
            c1 = layers.Conv1D(self.config['filters'], kernel_size=8, strides=stride_size, padding=pad,
                               activation=c_act)(c)
            c2 = layers.Conv1D(self.config['filters'], kernel_size=32, strides=stride_size, padding=pad,
                               activation=c_act)(c)

            c = layers.concatenate([c0, c1, c2])
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

        else:   # No inception Modules
            c = layers.Conv1D(self.config['filters'], kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(self.input)
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

            c = layers.Conv1D(self.config['filters'], kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

            c = layers.Conv1D(self.config['filters'], kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

        if res: # Residual RNN
            g1 = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                            dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(c)
            g2 = layers.GRU(self.config['state_size'], return_sequences=True,  activation=rk_act, recurrent_activation=r_act,
                            dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g1)
            g_concat1 = layers.concatenate([g1, g2])

            g3 = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                            dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g_concat1)
            g_concat2 = layers.concatenate([g1, g2, g3])

            g = layers.GRU(self.config['state_size'], return_sequences=False, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g_concat2)

        else:   # No Residual RNN
            g = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(c)

            g = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g)
            g = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g)

            g = layers.GRU(self.config['state_size'], return_sequences=False, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g)


        d = layers.Dense(self.config['output_size'])(g)
        out = layers.Softmax()(d)

        self.model = Model(self.input, out)
        print("{} initialized.".format(self.model.name))



    # -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- #
    def train(self, X, y, Xv=None, yv=None, verbose=1):
        print("Training {}".format(self.model.name))

        if Xv is None or yv is None:
            self.model.compile(loss=kr.losses.categorical_crossentropy,
                               optimizer=kr.optimizers.Adam(self.config['learning_rate']), metrics=['acc'])
            history = self.model.fit(x=X, y=y, batch_size=self.config['batch_size'],
                                     epochs=self.config['epochs'], verbose=verbose, validation_split=0.2, shuffle=True)

        else:
            self.model.compile(loss=kr.losses.categorical_crossentropy, optimizer=kr.optimizers.Adam(self.config['learning_rate']), metrics=['acc'])
            history = self.model.fit(x=X, y=y ,batch_size=self.config['batch_size'],
                                     epochs=self.config['epochs'], verbose=verbose,
                                    validation_data=(Xv, yv), shuffle=True)
        return history


    def predict_score(self, X):
        pred = self.model.predict(X)
        pred = np.argmax(pred, axis=1)  #TODO axis

        return pred


    def eval_error(self, X, y):
        pred = self.predict_score(X)

        return np.average(np.not_equal(np.argmax(y, axis=1), pred))


    def eval_acc(self, X, y):
        pred = self.predict_score(X)
        return np.average(np.equal(np.argmax(y, axis=1), pred))


    def save_model(self, name, path="model/"):
        self.model.save(path+name)

    def load_model(self, name, path="model/"):
        self.model = load_model(path+name)