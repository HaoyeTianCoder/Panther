from config_default_old import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from keras.models import Sequential
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC, LinearSVC

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import *
from keras.layers import Dense
from keras.layers import LSTM, Dense, Dropout

import tensorflow as tf
from tensorflow.keras.experimental import LinearModel, WideDeepModel
from keras.layers import Dense, Embedding, Dropout, Input, Concatenate

class Prediction:
    def __init__(self, dataset, labels, feature1_length, algorithm, split_method, kfold=10, ):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm
        self.split_method = split_method
        self.kfold = kfold
        self.feature1_length = feature1_length

    def evaluation_metrics(self, y_true, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)

        # minn = 1
        # for i in range(len(y_true)):
        #     if y_true[i] == 1:
        #         if y_pred_prob[i] < minn:
        #             minn = y_pred_prob[i]
        # print(minn)
        # y_pred = [1 if p >= (minn-0.000001) else 0 for p in y_pred_prob]
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # recall_p = tp / (tp + fn)
        # recall_n = tn / (tn + fp)
        # print('Thresh: {:.2f}, AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(minn, auc_, recall_p, recall_n))
        # for i in range(0,1001):
        #     threshold = i*0.001
        #     y_pred = [1 if p >= threshold else 0 for p in y_pred_prob]
        #     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #     recall_p = tp / (tp + fn)
        #     recall_n = tn / (tn + fp)
        #     print('Thresh: {:.2f}, AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(threshold, auc_, recall_p, recall_n))


        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_
        return auc_, recall_p, recall_n, acc, prc, rc, f1

    def run_cvfold(self, ):

        # scaler = StandardScaler()
        # scaler.fit_transform(self.dataset)

        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        for train_index, test_index in skf.split(self.dataset, self.labels):
            x_train, y_train = self.dataset[train_index], self.labels[train_index]
            x_test, y_test = self.dataset[test_index], self.labels[test_index]

            # standard data
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            clf = None
            if self.algorithm == 'lr':
                clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'dt':
                clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
            elif self.algorithm == 'rf':
                clf = RandomForestClassifier(n_estimators=100, ).fit(X=x_train, y=y_train)
            elif self.algorithm == 'xgb':
                dtrain = xgb.DMatrix(x_train, label=y_train)
                # params = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic', 'gamma': 1.0,
                #          'lambda': 1.0, 'subsample': 1}
                params = {'objective': 'binary:logistic', }
                num_boost_round = 10
                clf = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
            elif self.algorithm == 'lgb':
                x_train_lgb_dmatrix = lgb.Dataset(data=x_train, label=y_train)
                params = {'objective': 'binary', }
                num_boost_round = 100
                clf = lgb.train(params=params, train_set=x_train_lgb_dmatrix, num_boost_round=num_boost_round,)
            elif self.algorithm == 'svm':
                clf = SVC(gamma='auto', probability=True,).fit(X=x_train, y=y_train)
            elif self.algorithm == 'nb':
                clf = GaussianNB().fit(X=x_train, y=y_train)
            elif self.algorithm == 'xgb_xgb':
                x_train_xgb1 = x_train[:,:self.feature1_length]
                x_train_xgb1_dmatrix = xgb.DMatrix(x_train_xgb1, label=y_train)


                x_train_xgb2 = x_train[:,self.feature1_length:]
                x_train_xgb2_dmatrix = xgb.DMatrix(x_train_xgb2, label=y_train)

                params = {'objective': 'binary:logistic', }
                num_boost_round=10
                xgb1 = xgb.train(params=params, dtrain=x_train_xgb1_dmatrix, num_boost_round=num_boost_round)
                xgb2 = xgb.train(params=params, dtrain=x_train_xgb2_dmatrix, num_boost_round=num_boost_round)

            elif self.algorithm == 'lr_rf':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_rf = x_train[:,self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)
                rf = RandomForestClassifier().fit(X=x_train_rf, y=y_train)
            elif self.algorithm == 'svm_xgb':
                x_train_svm = x_train[:,:self.feature1_length:]
                x_train_xgb = x_train[:,self.feature1_length:]
                x_train_xgb_dmatrix = xgb.DMatrix(x_train_xgb, label=y_train)
                svm = SVC(gamma='auto', probability=True,).fit(X=x_train_svm, y=y_train)

                params = {'objective': 'binary:logistic',}
                num_boost_round = 10
                xgb_model = xgb.train(params=params, dtrain=x_train_xgb_dmatrix, num_boost_round=num_boost_round)
            elif self.algorithm == 'lr_xgb':
                x_train_lr = x_train[:,:self.feature1_length:]
                x_train_xgb = x_train[:,self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

                x_train_xgb_dmatrix = xgb.DMatrix(x_train_xgb, label=y_train)
                # params = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic', 'gamma': 1.0,
                #          'lambda': 1.0,
                #          'subsample': 1}
                params = {'objective': 'binary:logistic',}
                num_boost_round = 10
                xgb_model = xgb.train(params=params, dtrain=x_train_xgb_dmatrix, num_boost_round=num_boost_round)

                # # auto weight assign
                # lr_out = lr.predict_proba(x_train_lr)[:, 1].reshape((-1,1))
                # xgb_out = xgb_model.predict(x_train_xgb_dmatrix).reshape((-1,1))
                # data_label = np.concatenate((lr_out, xgb_out),axis=1)
                # weights = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=data_label, y=y_train).coef_
                #
                # weight_lr = weights[0,0]
                # weight_xgb = weights[0,1]
                # total = weight_lr + weight_xgb
                #
                # weight_lr = weight_lr/(total)
                # weight_xgb = weight_xgb/(total)

            elif self.algorithm == 'lr_lgb':
                x_train_lr = x_train[:, :self.feature1_length]
                x_train_lgb = x_train[:, self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train_lr, y=y_train)

                x_train_lgb_dmatrix = lgb.Dataset(data=x_train_lgb, label=y_train)

                params = {'objective': 'binary', }
                num_boost_round = 100
                lgb_model = lgb.train(params=params, train_set=x_train_lgb_dmatrix, num_boost_round=num_boost_round,)

            elif self.algorithm == 'lstm':
                # reshape input to be [samples, time steps, features]
                x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
                x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

                # create and fit the LSTM network
                clf = Sequential()
                clf.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
                clf.add(Dropout(0.4))
                clf.add(LSTM(128))
                clf.add(Dense(64, activation='relu'))
                clf.add(Dropout(0.2))
                clf.add(Dense(2, activation='softmax'))
                clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                clf.fit(x_train, y_train, epochs=50, batch_size=1000, verbose=2)

            # prediction
            if self.algorithm == 'lgb':
                x_test_lgb = x_test
                y_pred = clf.predict(x_test_lgb)
            elif self.algorithm == 'xgb':
                x_test_xgb = x_test
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
                y_pred = clf.predict(x_test_xgb_dmatrix)
            elif self.algorithm == 'svm_xgb':
                x_test_svm = x_test[:,:self.feature1_length]
                x_test_xgb = x_test[:,self.feature1_length:]
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)

                y_pred_svm = svm.predict_proba(x_test_svm)[:, 1]
                y_pred_xgb = xgb_model.predict(x_test_xgb_dmatrix)

                y_pred = y_pred_svm * 0.5 + y_pred_xgb * 0.5
            elif self.algorithm == 'xgb_xgb':
                x_test_xgb1 = x_test[:,:self.feature1_length]
                x_test_xgb1_dmatrix = xgb.DMatrix(x_test_xgb1, label=y_test)

                x_test_xgb2 = x_test[:,self.feature1_length:]
                x_test_xgb2_dmatrix = xgb.DMatrix(x_test_xgb2, label=y_test)

                y_pred_xgb1 = xgb1.predict(x_test_xgb1_dmatrix)
                y_pred_xgb2 = xgb2.predict(x_test_xgb2_dmatrix)

                # assign weight
                y_pred = y_pred_xgb1 * 0.5 + y_pred_xgb2 * 0.5

                # y_pred_xgb1 = list(y_pred_xgb1)
                # y_pred_xgb2 = list(y_pred_xgb2)
                # y_pred = []
                # for i in range(len(y_pred_xgb1)):
                #     if abs(y_pred_xgb1[i]-y_pred_xgb2[i]) >= 0.4:
                #         prob = y_pred_xgb1[i]
                #     else:
                #         prob = y_pred_xgb1[i] * 0.5 + y_pred_xgb2[i] * 0.5
                #     y_pred.append(prob)
            elif self.algorithm == 'lr_rf':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_rf = x_test[:,self.feature1_length:]

                y_pred_lr = lr.predict_proba(x_test_lr)[:, 1]
                y_pred_rf = rf.predict(x_test_rf)

                # assign weight
                y_pred = y_pred_lr * 0.50 + y_pred_rf * 0.50
            elif self.algorithm == 'lr_xgb':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_xgb = x_test[:,self.feature1_length:]
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)

                y_pred_lr = lr.predict_proba(x_test_lr)[:, 1]
                y_pred_xgb = xgb_model.predict(x_test_xgb_dmatrix)

                # assign weight
                y_pred = y_pred_lr * 0.50 + y_pred_xgb * 0.50
                # y_pred = y_pred_lr * weight_lr + y_pred_xgb * weight_xgb
            else:
                y_pred = clf.predict_proba(x_test)[:, 1]
            auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

            # # calculate SHAP value
            if self.algorithm == 'lr_lgb':
                # lgb SHAP
                explainer_tree = shap.TreeExplainer(lgb_model)
                shap_values_tree = explainer_tree.shap_values(x_train_lgb, )
                shap.summary_plot(shap_values_tree, x_train_lgb)
                shap.summary_plot(shap_values_tree, x_train_lgb, plot_type="bar")

                # lr SHAP
                explainer_lr = shap.LinearExplainer(lr, x_train_lr)
                shap_values_lr = explainer_lr.shap_values(x_train_lr)
                shap.summary_plot(shap_values_lr, x_train_lr)
                shap.summary_plot(shap_values_lr, x_train_lr, plot_type="bar")
            elif self.algorithm == 'lr_xgb':
                # xgb SHAP
                explainer_tree = shap.TreeExplainer(xgb_model)
                shap_values_tree = explainer_tree.shap_values(x_test_xgb, )
                shap.summary_plot(shap_values_tree, x_test_xgb, max_display=20)
                shap.summary_plot(shap_values_tree, x_test_xgb, plot_type="bar")

                # lr SHAP
                explainer_lr = shap.LinearExplainer(lr, x_train_lr)
                shap_values_lr = explainer_lr.shap_values(x_test_lr)
                shap.summary_plot(shap_values_lr, x_test_lr)
                shap.summary_plot(shap_values_lr, x_test_lr, plot_type="bar")

        # print('------------------------------------------------------------------------')
        # print('{}-fold cross validation'.format(self.kfold))
        print('')
        print('{}-fold cross validation mean: '.format(self.kfold))
        print('Accuracy: {:.3f} -- Precision: {:.3f} -- +Recall: {:.3f} -- F1: {:.3f} -- AUC: {:.3f}'.format(np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean()))

        output2 = '------------------------------------\n'
        output2 += '{}-fold cross validation\n'.format(self.kfold)
        output2 += 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}\n\n'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean())

        return output2

    def run_slice(self, ):

        scaler = StandardScaler()
        scaler.fit_transform(self.dataset)

        # skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        correct_index = np.where(self.labels == 1)[0][:20]
        incorrect_index = np.where(self.labels == 0)[0]


        for i in range(self.kfold):
            incorrect_index_tmp = np.array(random.sample(list(incorrect_index), correct_index.shape[0]))
            new_index = np.concatenate((correct_index, incorrect_index_tmp),axis=0)
            new_data = self.dataset[new_index]
            new_label = self.labels[new_index]

            x_train, x_test, y_train, y_test = train_test_split(new_data, new_label, test_size=0.2)

            # x_train, y_train = self.dataset[train_index], self.labels[train_index]
            # x_test, y_test = self.dataset[test_index], self.labels[test_index]

            clf = None
            if self.algorithm == 'lr':
                clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'rf':
                clf = RandomForestClassifier(class_weight={1: 1}, n_estimators=1000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'lr_rf':
                x_train_lr = x_train[:,:self.feature1_length]
                x_train_rf = x_train[:,self.feature1_length:]

                lr = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train_lr, y=y_train)
                rf = RandomForestClassifier(class_weight={1: 1}, n_estimators=1000).fit(X=x_train_rf, y=y_train)


            # prediction
            if self.algorithm == 'lr_rf':
                x_test_lr = x_test[:,:self.feature1_length]
                x_test_rf = x_test[:,self.feature1_length:]

                y_pred_lr = lr.predict_proba(x_test_lr)[:, 1]
                y_pred_rf = rf.predict_proba(x_test_rf)[:, 1]

                # assign weight
                y_pred = y_pred_lr * 0.6 + y_pred_rf * 0.4
            else:
                y_pred = clf.predict_proba(x_test)[:, 1]
            auc_, recall_p, recall_n = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

            # accs.append(acc)
            # prcs.append(prc)
            # rcs.append(rc)
            # f1s.append(f1)

            print(clf.coef_)
            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

        # print('------------------------------------------------------------------------')
        # print('{}-fold cross validation'.format(self.kfold))
        # print('Accuracy: {} -- Precision: {} -- +Recall: {} -- F1: {} -- AUC: {}'.format(np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))
        print('')
        print('{}-fold cross validation mean: '.format(self.kfold))

        output2 = '------------------------------------\n'
        output2 += '{}-time slice validation\n'.format(self.kfold)
        output2 += 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean())

        return output2