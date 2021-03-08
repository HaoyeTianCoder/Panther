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


def evaluation_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall_p =  tp / (tp + fn)
    recall_n = tn / (tn + fp)
    print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
    # return acc, prc, rc, f1, auc_
    return auc_, recall_p, recall_n, acc, prc, rc, f1

# eng = StandardScaler()
# ber = StandardScaler()
# data_engineered = eng.fit_transform(data_engineered)
# data_bert = ber.fit_transform(data_bert)
def get_dnn():
    input_embeddings_tensor = Input(shape=(2050,))
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

def get_dnn_4engineered():
    input_embeddings_tensor = Input(shape=(4497,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(1024, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    # embeddings_tensor = Dense(64, activation='tanh')(embeddings_tensor)
    output_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    model = models.Model(inputs=input_embeddings_tensor, outputs=output_tensor)
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

def get_dnn_cnn():
    input_embeddings_tensor = Input(shape=(2050,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)

    # embeddings_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    input_engineered_tensor = Input(shape=(4497,))
    # CNN
    engineered_tensor = Reshape((4497, 1))(input_engineered_tensor)
    engineered_tensor = Conv1D(64, (200), activation='relu')(engineered_tensor)
    engineered_tensor = MaxPool1D((8))(engineered_tensor)
    engineered_tensor = Conv1D(64, (200), activation='relu')(engineered_tensor)
    engineered_tensor = MaxPool1D((4))(engineered_tensor)
    engineered_tensor = Flatten()(engineered_tensor)

    # engineered_tensor = Dense(1, activation='sigmoid')(engineered_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, engineered_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_engineered_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    plot_model(model, to_file='./model.png', show_shapes=True)

    return model

def get_dnn_dnn():
    input_embeddings_tensor = Input(shape=(2050,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)

    # embeddings_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    input_engineered_tensor = Input(shape=(4497,))
    engineered_tensor = Dense(1024, activation='tanh')(input_engineered_tensor)
    engineered_tensor = Dense(1024, activation='tanh')(engineered_tensor)
    engineered_tensor = Dense(512, activation='tanh')(engineered_tensor)

    engineered_tensor = Dense(128, activation='sigmoid')(engineered_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, engineered_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_engineered_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    plot_model(model, to_file='./model.png', show_shapes=True)

    return model

def get_wide_deep():
    input_embeddings_tensor = Input(shape=(2050,))
    embeddings_tensor = Dense(1024, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(512, activation='tanh')(embeddings_tensor)
    # embeddings_tensor = Dense(64, activation='tanh')(embeddings_tensor)

    input_fe_tensor = Input(shape=(4497,))
    # engineered_tensor = Dense(1, activation='sigmoid')(input_fe_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, input_fe_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_fe_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    plot_model(model, to_file='./model.png', show_shapes=True)

    return model

if __name__ == '__main__':
    # 修改数据路径
    data_engineered = np.load('data_vector/engineered_ods.npy')
    data_bert = np.load('data_vector/learned_Bert.npy')
    labels = np.load('data_vector/labels.npy')

    # 样本
    print('data_engineered', data_engineered.shape)
    print('data_bert', data_bert.shape)
    print('labels', labels.shape)

    print('正样本', sum(labels))
    print('负样本', len(labels) - sum(labels))

    fold = 10
    skf = StratifiedKFold(n_splits=fold, shuffle=True)
    accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
    rcs_p, rcs_n = list(), list()
    for train_index, test_index in skf.split(data_engineered, labels):
        x_train_data_engineered = data_engineered[train_index]
        x_train_data_bert = data_bert[train_index]
        y_train = labels[train_index]

        x_test_data_engineered = data_engineered[test_index]
        x_test_data_bert = data_bert[test_index]
        y_test = labels[test_index]

        # standard engineered data
        scaler_e = StandardScaler().fit(x_train_data_engineered)
        x_train_data_engineered = scaler_e.transform(x_train_data_engineered)
        x_test_data_engineered = scaler_e.transform(x_test_data_engineered)

        # standard bert data
        scaler_b = StandardScaler().fit(x_train_data_bert)
        x_train_data_bert = scaler_b.transform(x_train_data_bert)
        x_test_data_bert = scaler_b.transform(x_test_data_bert)

        # 1 get lr
        # model = get_cnn()
        # callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1),]
        # model.fit(x_train_data_engineered, y_train, callbacks=callback, batch_size=32, epochs=10, )
        # y_pred = model.predict(x_test_data_engineered,)[:, 0]

        # 2 get dnn
        # # model = get_dnn()
        # model = get_dnn_4engineered()
        # callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1), keras.callbacks.ModelCheckpoint(filepath='./model/best_model.h5', save_best_only=True)]
        # # model.fit(x_train_data_bert, y_train, batch_size=32 ,epochs=10, )
        # model.fit(x_train_data_engineered, y_train, callbacks=callback, batch_size=32 ,epochs=10, )
        # # y_pred = model.predict(x_test_data_bert,)[:, 0]
        # y_pred = model.predict(x_test_data_engineered,)[:, 0]

        # # 3 get combine
        model = get_wide_deep()
        callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1),]
        model.fit([x_train_data_bert, x_train_data_engineered], y_train, callbacks=callback, batch_size=32 ,epochs=10, )
        y_pred = model.predict([x_test_data_bert, x_test_data_engineered])[:, 0]

        # model = get_lr_cnn()
        # callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=1, mode="max", verbose=1), keras.callbacks.ModelCheckpoint(filepath='./model/best_model.h5', save_best_only=True)]
        # # model.fit([x_train_data_bert, x_train_data_engineered], y_train, callbacks=callback, validation_data=([x_test_data_bert, x_test_data_engineered], y_test), batch_size=32 ,epochs=10, )
        # model.fit([x_train_data_bert, x_train_data_engineered], y_train, callbacks=callback, validation_split=0.1, batch_size=32 ,epochs=10, )
        # # saved_model = load_model('./model/best_model.h5')
        # y_pred = model.predict([x_test_data_bert, x_test_data_engineered])[:, 0]

        auc_, recall_p, recall_n, acc, prc, rc, f1 = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

        accs.append(acc)
        prcs.append(prc)
        rcs.append(rc)
        f1s.append(f1)

        aucs.append(auc_)
        rcs_p.append(recall_p)
        rcs_n.append(recall_n)
        # print(auc_, recall_p, recall_n)

        # for i in range(1, 10):
        #     y_pred_tn = [1 if p >= i / 10.0 else 0 for p in y_pred]
        #     tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
        #     print('i:{}'.format(i / 10), end=' ')
        #     print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))

    print('{} fold: '.format(fold))

    print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100, np.array(f1s).mean() * 100, np.array(aucs).mean()))
    output2 = 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean())
    print(output2)




