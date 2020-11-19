from config_default_old import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn import *
from sklearn.metrics import confusion_matrix
import random
from sklearn.model_selection import train_test_split

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
        # acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        # prc = precision_score(y_true=y_true, y_pred=y_pred)
        # rc = recall_score(y_true=y_true, y_pred=y_pred)
        # f1 = 2 * prc * rc / (prc + rc)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p =  tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        # return acc, prc, rc, f1, auc_
        return auc_, recall_p, recall_n

    def run_cvfold(self, ):

        scaler = StandardScaler()
        scaler.fit_transform(self.dataset)

        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()

        for train_index, test_index in skf.split(self.dataset, self.labels):
            x_train, y_train = self.dataset[train_index], self.labels[train_index]
            x_test, y_test = self.dataset[test_index], self.labels[test_index]
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

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

        # print('------------------------------------------------------------------------')
        # print('{}-fold cross validation'.format(self.kfold))
        # print('Accuracy: {} -- Precision: {} -- +Recall: {} -- F1: {} -- AUC: {}'.format(np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))
        print('')
        print('{}-fold cross validation mean: '.format(self.kfold))
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean()))

        output2 = '------------------------------------\n'
        output2 += '{}-fold cross validation\n'.format(self.kfold)
        output2 += 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean())

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