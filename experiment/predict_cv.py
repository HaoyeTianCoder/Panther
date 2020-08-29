from config_default import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score


class Prediction:
    def __init__(self, dataset, labels, algorithm, kfold=10, ):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm
        self.kfold = kfold
    def evaluation_metrics(self, y_true, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)
        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        return acc, prc, rc, f1, auc_

    def run(self, ):
        scaler = StandardScaler()
        scaler.fit_transform(self.dataset)

        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        for train_index, test_index in skf.split(self.dataset, self.labels):
            x_train, y_train = self.dataset[train_index], self.labels[train_index]
            x_test, y_test = self.dataset[test_index], self.labels[test_index]
            clf = None
            if self.algorithm == 'lr':
                clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'rf':
                clf = RandomForestClassifier(class_weight={1: 1}, n_estimators=1000).fit(X=x_train, y=y_train)

            y_pred = clf.predict_proba(x_test)[:, 1]
            acc, prc, rc, f1, auc_ = self.evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)
            aucs.append(auc_)
        print('------------------------------------------------------------------------')
        print('{}-fold cross validation'.format(self.kfold))
        print('Accuracy: {} -- Precision: {} -- +Recall: {} -- F1: {} -- AUC: {}'.format(
        np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))