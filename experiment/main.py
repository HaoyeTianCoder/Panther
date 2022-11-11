# from config_default import *
import numpy as np

import predict_cv, train_doc
from preprocess import deduplicate, obtain_ods_feature, save_feature
import pandas as pd
import argparse
import sys
sys.path.append("..")
from config_default import *
import pickle
import random, math
from statsmodels.stats.contingency_tables import mcnemar


class Experiment:
    def __init__(self, fea_used, w2v, path_data_pickle, split_method, algorithm, explanation=None):
        self.fea_used = fea_used
        self.w2v = w2v
        self.split_method = split_method
        self.algorithm = algorithm
        self.explanation = explanation
        self.feature1_length = None
        self.result = ''

        self.path_learned_feature = ''
        self.path_engineered_feature = ''
        self.path_labels = ''
        self.path_record = ''


        self.path_data_pickle = path_data_pickle
        self.dataset_json = ''
        
    def load_combine_data(self, ):
        # load data
        if not os.path.exists(self.path_data_pickle):
            # logging.info('miss the path of the datset ......')
            raise Exception('miss the path of the datset: {} ......'.format(self.path_learned_feature))
        output = '------------------------------------\n'
        output += 'Loading dataset:\n'

        with open(path_data_pickle, 'rb') as f:
            self.dataset_json = pickle.load(f)


    def combine_feature(self, path_learned_feature, path_engineered_feature):
        learned_feature = np.load(path_learned_feature)
        engineered_feature = np.load(path_engineered_feature)

        dataset = np.concatenate((learned_feature, engineered_feature), axis=1)

        self.dataset = dataset
        self.feature1_length = learned_feature.shape[1]

    def train_predict(self, ):
        output1 = '------------------------------------\n'
        output1 += 'Experiment design: \n'
        output1 += 'Feature used: {}. W2V: {}. ML_algorithm: {}\n'.format(self.fea_used, self.w2v if self.fea_used=='learned' else '', self.algorithm)
        output1 += '------------------------------------\n'
        # output1 += 'Result: \n'

        print(output1, end='')
        self.result += output1

        # 10 group cross validation
        times = 10
        dataset_distribution = []
        metrics_y_true = []
        metrics_learned = []
        metrics_engineered = []
        metrics_combine = []
        metrics_patchsim = []
        learned_correctness_all, engineered_correctness_all, combine_correctness_all, learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all = list(), list(), list(), list(), list(), list()
        project_ids = list(self.dataset_json.keys())
        random.seed(100)
        random.shuffle(project_ids,)
        n = int(math.ceil(len(project_ids) / float(times)))
        groups = [project_ids[i:i+n] for i in range(0, len(project_ids), n)]

        all_labels = []
        for train_id in project_ids:
            value = self.dataset_json[train_id]
            for p in range(len(value)):
                label = value[p][3]
                all_labels.append(label)
        print('Dataset size: {}, Incorrect: {}, Correct: {}'.format(len(all_labels), all_labels.count(0),
                                                                    all_labels.count(1)))
        print('Algorithm: {}'.format(algorithm))
        print('#####')

        for i in range(times):
            test_group = groups[i]
            train_group = groups[:i] + groups[i+1:]

            test_ids = test_group
            train_ids = []
            for j in train_group:
                train_ids += j

            train_features_learned, train_features_engineered, train_features_combine, train_labels, train_infos, _ = self.get_feature(train_ids, self.dataset_json)
            test_features_learned, test_features_engineered, test_features_combine, test_labels, test_infos, y_patchsim_one = self.get_feature(test_ids, self.dataset_json)
            # labels = train_labels + test_labels
            print('Train data size: {}, Incorrect: {}, Correct: {}'.format(len(train_labels), list(train_labels).count(0), list(train_labels).count(1)))
            print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(test_labels), list(test_labels).count(0), list(test_labels).count(1)))
            print('#####')
            dataset_distribution.append([len(train_labels), len(test_labels)])


            # init prediction
            pd = predict_cv.Prediction(train_features_learned, train_features_engineered, train_features_combine, train_labels, \
                                        test_features_learned, test_features_engineered, test_features_combine, test_labels, test_infos, self.algorithm, split_method, fea_used)
            pd.explanation = self.explanation

            pd.run_cvgroup(i)

            metrics_y_true += pd.y_true_one
            metrics_learned += pd.y_pred_l_one
            metrics_engineered += pd.y_pred_e_one
            if fea_used == 'combine':
                metrics_combine += pd.y_pred_c_one
            if self.split_method == 'compare4patchsim':
                metrics_patchsim += list(y_patchsim_one)

            if fea_used == 'combine':
                learned_correctness_all += pd.result_venn[0]
                engineered_correctness_all += pd.result_venn[1]
                combine_correctness_all += pd.result_venn[2]

                learned_incorrectness_all += pd.result_venn[3]
                engineered_incorrectness_all += pd.result_venn[4]
                combine_incorrectness_all += pd.result_venn[5]


        # auc curve
        # pd.draw_auc(y_pmetrics_learnedred, metrics_y_true)

        # confusion matrix
        # pd.confusion_matrix(metrics_learned, metrics_y_true)

        if split_method == 'cvgroup':
            print('Learned feature:')
            pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_learned,)
            print('Engineering feature:')
            t = 0.43 if algorithm in {'xgb', 'ensemble_xgb','naive_xgb'} else 0.5
            pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_engineered, t=t)
            if fea_used == 'combine':
                print('Combined feature:')
                pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_combine)
                self.mcnemar_test(metrics_y_true, metrics_learned, metrics_engineered, metrics_combine)

            self.venn4feature(metrics_learned, metrics_engineered, metrics_y_true, pd, t=0.60)

        elif split_method == 'compare4patchsim':
            print('Learned feature:')
            # pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_learned, t=0.21) # for random forest
            pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_learned, t=0.01) # for xgb
            # pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_learned, t=0.1) # for dnn
            print('Engineering feature:')
            # t= 0.43 if algorithm = 'ensemble_xgb' or algorithm = 'naive_xgb'
            pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_engineered)
            if fea_used == 'combine':
                print('Combined feature:')
                pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_combine)
            print('Patchsim:')
            pd.evaluation_metrics(y_true=metrics_y_true, y_pred_prob=metrics_patchsim)

            self.venn4feature(metrics_learned, metrics_patchsim, metrics_y_true, pd, t=0.01)


        # metrics_learned = np.round(np.array(metrics_learned).mean(axis=0,),3)
        # metrics_engineered = np.round(np.array(metrics_engineered).mean(axis=0),3)
        # metrics_combine = np.round(np.array(metrics_combine).mean(axis=0),3)
        # print('auc_l:{}, recall_p_l:{}, recall_n_l:{}, acc_l:{}, prc_l:{}, rc_l:{}, f1_l:{}'.format(metrics_learned[0], metrics_learned[1], metrics_learned[2], metrics_learned[3], metrics_learned[4], metrics_learned[5], metrics_learned[6]))
        # print('auc_e:{}, recall_p_e:{}, recall_n_e:{}, acc_e:{}, prc_e:{}, rc_e:{}, f1_e:{}'.format(metrics_engineered[0], metrics_engineered[1], metrics_engineered[2], metrics_engineered[3], metrics_engineered[4], metrics_engineered[5], metrics_engineered[6]))
        # print('auc_c:{}, recall_p_c:{}, recall_n_c:{}, acc_c:{}, prc_c:{}, rc_c:{}, f1_c:{}'.format(metrics_combine[0], metrics_combine[1], metrics_combine[2], metrics_combine[3], metrics_combine[4], metrics_combine[5], metrics_combine[6]))

        if fea_used == 'combine' and split_method == 'cvgroup':
        # if self.algorithm == 'xgb_xgb' or self.algorithm == 'lr_xgb' or self.algorithm == 'dnn_dnn_venn' or self.algorithm == 'xgb_combine' or self.algorithm == 'lr_combine':
        #     pd.draw_venn2(learned_correctness_all, combine_correctness_all, learned_incorrectness_all, combine_incorrectness_all)
            pd.draw_venn3(learned_correctness_all, engineered_correctness_all, combine_correctness_all, learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all)

            # # correct included
            independent_learned, independent_engineered, independent_combine = pd.independent_case(learned_correctness_all, engineered_correctness_all, combine_correctness_all)
            # print('Correct Included: ')
            # print('independent_learned: {}'.format(independent_learned))
            # print('independent_engineered: {}'.format(independent_engineered))
            # print('independent_combine: {}'.format(independent_combine))
            # # incorrect excluded
            # independent_learned, independent_engineered, independent_combine = self.independent_case(learned_incorrectness_all, engineered_incorrectness_all, combine_incorrectness_all)
            # print('Correct Included: ')
            # print('independent_learned: {}'.format(independent_learned))
            # print('independent_engineered: {}'.format(independent_engineered))
            # print('independent_combine: {}'.format(independent_combine))

    def get_feature(self, ids, dataset_json, ):
        # patchsim test data
        patchsim_name = {}
        y_patchsim_one = []
        if self.split_method == 'compare4patchsim':

            with open('patchsim_result.txt', 'r+') as f:
                for line in f:
                    name = line.split(',')[0][:-2]
                    label = int(line.split(',')[1])
                    # patchsim_name.append(name)
                    # y_patchsim.append(label)
                    patchsim_name[name] = label

        features_learned, features_engineered, features_combine, labels, infos = [], [], [], [], []
        for id in ids:
            list_patches = dataset_json[id]
            for patch_tuple in list_patches:
                info = patch_tuple[0]

                if self.split_method == 'compare4patchsim':
                    if info[:-2] not in patchsim_name:
                        continue
                    else:
                        y_patchsim = patchsim_name[info[:-2]]
                        y_patchsim_one.append(y_patchsim)

                        learned_feature = patch_tuple[1]
                        engineered_feature = patch_tuple[2]
                        label = patch_tuple[3]

                        features_learned.append(learned_feature)
                        features_engineered.append(engineered_feature)
                        feature_combine = np.append(learned_feature, engineered_feature)
                        features_combine.append(feature_combine)

                        labels.append(label)
                        infos.append(info)
                else:
                    learned_feature = patch_tuple[1]
                    engineered_feature = patch_tuple[2]
                    label = patch_tuple[3]

                    features_learned.append(learned_feature)
                    features_engineered.append(engineered_feature)
                    feature_combine = np.append(learned_feature, engineered_feature)
                    features_combine.append(feature_combine)

                    labels.append(label)
                    infos.append(info)
        return np.array(features_learned), np.array(features_engineered), np.array(features_combine), np.array(labels), infos, y_patchsim_one

    def mcnemar_test(self, metrics_y_true, metrics_learned, metrics_engineered, metrics_combine):
        metrics_learned = [1 if p >= 0.5 else 0 for p in metrics_learned]
        metrics_engineered = [1 if p >= 0.5 else 0 for p in metrics_engineered]
        metrics_combine = [1 if p >= 0.5 else 0 for p in metrics_combine]

        yesyes_learned, yesno_learned, noyes_learned, nono_learned = 0, 0, 0, 0
        for i in range(len(metrics_y_true)):
            if metrics_y_true[i] == metrics_combine[i] and metrics_y_true[i] == metrics_engineered[i]:
                yesyes_learned += 1
            elif metrics_y_true[i] == metrics_combine[i] and metrics_y_true[i] != metrics_engineered[i]:
                yesno_learned += 1
            elif metrics_y_true[i] != metrics_combine[i] and metrics_y_true[i] == metrics_engineered[i]:
                noyes_learned += 1
            elif metrics_y_true[i] != metrics_combine[i] and metrics_y_true[i] != metrics_engineered[i]:
                nono_learned += 1

        yesyes_engineered, yesno_engineered, noyes_engineered, nono_engineered = 0, 0, 0, 0
        for i in range(len(metrics_y_true)):
            if metrics_y_true[i] == metrics_combine[i] and metrics_y_true[i] == metrics_learned[i]:
                yesyes_engineered += 1
            elif metrics_y_true[i] == metrics_combine[i] and metrics_y_true[i] != metrics_learned[i]:
                yesno_engineered += 1
            elif metrics_y_true[i] != metrics_combine[i] and metrics_y_true[i] == metrics_learned[i]:
                noyes_engineered += 1
            elif metrics_y_true[i] != metrics_combine[i] and metrics_y_true[i] != metrics_learned[i]:
                nono_engineered += 1

        # define contingency table
        table_learned = [[yesyes_learned, yesno_learned],
                        [noyes_learned, nono_learned]]
        table_engineered = [[yesyes_engineered, yesno_engineered],
                        [noyes_engineered, nono_engineered]]

        # calculate mcnemar test for learned
        result = mcnemar(table_learned, exact=False, correction=True)
        # summarize the finding
        print('statistic=%.3f, p-value=%.10f' % (result.statistic, result.pvalue))
        # interpret the p-value
        alpha = 0.05
        if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
        else:
            print('learned significant difference!')

        # calculate mcnemar test for engineered
        result = mcnemar(table_engineered, exact=False, correction=True)
        # summarize the finding
        print('statistic=%.3f, p-value=%.10f' % (result.statistic, result.pvalue))
        # interpret the p-value
        alpha = 0.05
        if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
        else:
            print('engineered significant difference!')

    def venn4feature(self, metrics_l_e, metrics_patchsim, metrics_y_true, pd, t=0.5):
        # learned feature: p for xgb
        y_pred_int = [1 if p >= t else 0 for p in metrics_l_e]
        metrics_patchsim = [1 if p >= 0.5 else 0 for p in metrics_patchsim]

        index_c = [i for i in range(len(list(metrics_y_true))) if list(metrics_y_true)[i] == 1]
        index_c_learned = [str(i) for i in index_c if y_pred_int[i] == 1]
        index_c_patchsim = [str(i) for i in index_c if metrics_patchsim[i] == 1]

        index_in = [i for i in range(len(list(metrics_y_true))) if list(metrics_y_true)[i] == 0]
        index_in_learned = [str(i) for i in index_in if y_pred_int[i] == 0]
        index_in_patchsim = [str(i) for i in index_in if metrics_patchsim[i] == 0]
        pd.draw_venn2(index_c_learned, index_c_patchsim, index_in_learned, index_in_patchsim)

    def run(self, ):
        # load single feature and decide whether combine
        self.load_combine_data()

        # split, train, predict
        self.train_predict()

        # save result
        self.save_result()


    def save_result(self):
        out_foler = '../result/'
        if not os.path.exists(out_foler):
            os.makedirs(out_foler)
        out_path = out_foler + self.fea_used + '.result'
        with open(out_path,'a+') as file:
            file.write(self.result)

# parser = argparse.ArgumentParser(description='Test for argparse')
# parser.add_argument('--w2v', '-w', help='word2vector',)
# parser.add_argument('--version', '-v', help='dataset verison', default='V1U')
# parser.add_argument('--path', '-p', help='absolute path of dataset', )
# parser.add_argument('--task', '-t', help='task', )
# args = parser.parse_args()

if __name__ == '__main__':
    # config
    cfg = Config()
    path_dataset = cfg.path_dataset
    path_testdata = ''
    version = cfg.version
    w2v = cfg.wcv

    if len(sys.argv) == 2:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = ''
        arg3 = ''
        arg4 = ''
    elif len(sys.argv) == 3:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = ''
        arg4 = ''
    elif len(sys.argv) == 4:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
        arg4 = ''
    elif len(sys.argv) == 5:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
        arg4 = sys.argv[4]
    else:
        arg1 = 'experiment'
        arg2 = 'cvgroup'
        arg3 = 'single'
        arg4 = 'xgb'

    task = arg1
    print('TASK: {}'.format(task))
    if task == 'deduplicate':
        # drop same patch
        if 'Unique' in path_dataset:
            print('already deduplicated!')
        else:
            dataset_name = path_dataset.split('/')[-1]
            path_dataset, dataset_name = deduplicate.deduplicate_by_token_with_location(dataset_name, path_dataset)

    # optional
    elif task == 'train_doc':
        path_dataset_all = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2'
        d = train_doc.doc(path_dataset_all)
        d.train()

    elif task == 'ods_feature':
        # generate ods feature json under folder where patch is
        obtain_ods_feature.obtain_ods_features(path_dataset)

    elif task == 'save_npy':
        # save learned feature and engineered feature to npy for prediction later
        other = 'ods'
        # save_feature.save_npy(path_dataset, w2v, other, version)
        save_feature.save_npy_bugids(path_dataset, w2v, other,)

    elif task == 'save_npy_4test':
        # for test data
        other = 'ods'
        # version is 'Cross_bigdata' or 'TestData'
        save_feature.save_npy_test(path_dataset, path_testdata, w2v, other, version)

    elif task == 'experiment':
        # start experiment
        print('version: {}  w2c: {}'.format(version, w2v))
        path_data_pickle = '../data/' + w2v + '.pickle'
        explanation = None

        # path_learned_feature = folder+'/learned_'+w2v+'.npy'
        # path_engineered_feature = folder+'/engineered_ods.npy'
        # path_labels = folder+'/labels.npy'
        # record = folder+'/record.txt'

        split_method = arg2
        fea_used = arg3
        algorithm = arg4

        if split_method == 'compare4patchsim':
            fea_used = 'single'
            algorithm = 'xgb'
        if split_method == 'SHAP':
            split_method = 'cvgroup'
            fea_used = 'combine'
            algorithm = 'naive_xgb'
            explanation = 'SHAP'

        # if fea_used == 'single':
        #     # algorithm = 'dt'
        #     # algorithm = 'rf'
        #     algorithm = 'xgb'
        #     # algorithm = 'dnn'
        # elif fea_used == 'combine':
        #     algorithm = 'ensemble_rf'
        #     # algorithm = 'naive_rf'
        #
        #     # algorithm = 'ensemble_xgb'
        #     # algorithm = 'naive_xgb'
        #
        #     # algorithm = 'deep_combine'

        e = Experiment(fea_used, w2v, path_data_pickle, split_method, algorithm, explanation)
        e.run()

        
        



