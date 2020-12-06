import random
import numpy as np
import lightgbm as lgb
import keras
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from keras.models import load_model

from sklearn import preprocessing
from keras import models
from keras.layers import MaxPool1D, Activation, Dense, Flatten, Input, Multiply, Permute, RepeatVector, Reshape, Concatenate, Conv1D
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score

def get_dnn(dimension):
    input_embeddings_tensor = Input(shape=(dimension,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    # embeddings_tensor = Dense(64, activation='tanh')(embeddings_tensor)
    output_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    model = models.Model(inputs=input_embeddings_tensor, outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='./model.png', show_shapes=True)
    return model

def get_dnn_4engineered(dimension):
    input_engineered_tensor = Input(shape=(dimension,))
    engineered_tensor = Dense(1024, activation='tanh')(input_engineered_tensor)
    # engineered_tensor = Dense(1024, activation='tanh')(engineered_tensor)
    engineered_tensor = Dense(1024, activation='tanh')(engineered_tensor)

    engineered_tensor = Dense(512, activation='sigmoid')(engineered_tensor)
    output_tensor = Dense(1, activation='sigmoid')(engineered_tensor)

    model = models.Model(inputs=input_engineered_tensor, outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='./model.png', show_shapes=True)
    return model

def get_cnn():
    input_engineered_tensor = Input(shape=(4497,))
    engineered_tensor = Reshape((4497, 1))(input_engineered_tensor)

    engineered_tensor = Conv1D(256, (50), activation='relu')(engineered_tensor)
    engineered_tensor = MaxPool1D((8))(engineered_tensor)
    # engineered_tensor = Conv1D(128, (50), activation='relu')(engineered_tensor)
    # engineered_tensor = MaxPool1D((8))(engineered_tensor)

    engineered_tensor = Flatten()(engineered_tensor)
    output_tensor = Dense(1, activation='sigmoid')(engineered_tensor)

    model = models.Model(inputs=input_engineered_tensor, outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='./model.png', show_shapes=True)
    return model

def get_wide_deep(dimension_learned, dimension_engineered):
    input_embeddings_tensor = Input(shape=(dimension_learned,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    # embeddings_tensor = Dense(64, activation='tanh')(embeddings_tensor)

    input_fe_tensor = Input(shape=(dimension_engineered,))
    # engineered_tensor = Dense(1, activation='sigmoid')(input_fe_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, input_fe_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_fe_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    plot_model(model, to_file='../model/wide_deep.png', show_shapes=True)

    return model

def get_dnn_dnn(dimension_learned, dimension_engineered):
    input_embeddings_tensor = Input(shape=(dimension_learned,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)

    # embeddings_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    input_engineered_tensor = Input(shape=(dimension_engineered,))
    engineered_tensor = Dense(1024, activation='tanh')(input_engineered_tensor)
    engineered_tensor = Dense(1024, activation='tanh')(engineered_tensor)
    engineered_tensor = Dense(512, activation='tanh')(engineered_tensor)

    # engineered_tensor = Dense(128, activation='sigmoid')(engineered_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, engineered_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_engineered_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    plot_model(model, to_file='../model/dnn_dnn.png', show_shapes=True)

    return model

def get_dnn_cnn(dimension_learned, dimension_engineered):
    input_embeddings_tensor = Input(shape=(dimension_learned,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)

    # embeddings_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    input_engineered_tensor = Input(shape=(dimension_engineered,))
    # CNN
    engineered_tensor = Reshape((dimension_engineered, 1))(input_engineered_tensor)
    engineered_tensor = Conv1D(32, (50), activation='relu')(engineered_tensor)
    engineered_tensor = MaxPool1D((2))(engineered_tensor)
    engineered_tensor = Conv1D(32, (50), activation='relu')(engineered_tensor)
    engineered_tensor = MaxPool1D((2))(engineered_tensor)
    engineered_tensor = Flatten()(engineered_tensor)

    # engineered_tensor = Dense(1, activation='sigmoid')(engineered_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, engineered_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_engineered_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    plot_model(model, to_file='../model/dnn_cnn.png', show_shapes=True)

    return model