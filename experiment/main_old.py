# from config_default import *
import predict_cv_old, train_doc
from preprocess import deduplicate, obtain_ods_feature, save_feature
import pandas as pd
import argparse
import sys
sys.path.append("..")
from config_default import *

class Experiment:
    def __init__(self, fea_used, w2v, path_learned_feature, path_engineered_feature, path_testdata, path_labels, record, split_method, algorithm, combine_method=None):
        self.fea_used = fea_used
        self.w2v = w2v
        self.combine_method = combine_method
        self.split_method = split_method
        self.algorithm = algorithm
        self.feature1_length = None
        self.result = ''

        self.path_learned_feature = path_learned_feature
        self.path_engineered_feature = path_engineered_feature
        self.path_testdata =path_testdata
        self.path_labels = path_labels
        self.path_record = record


        self.dataset = None
        self.record = None
        # self.learned_feature = None
        # self.engineered_feature = None
        self.label = None
        # self.embedding_length = embedding_length
        
    def load_combine_data(self, ):
        # load data
        if not os.path.exists(self.path_learned_feature) or not os.path.exists(self.path_engineered_feature) or not os.path.exists(self.path_labels):
            # logging.info('miss the path of the datset ......')
            raise Exception('miss the path of the datset: {} ......'.format(self.path_learned_feature))
        output = '------------------------------------\n'
        output += 'Loading dataset:\n'
        
        if fea_used == 'learned':
            self.dataset = np.load(self.path_learned_feature)
        elif fea_used == 'engineered':
            self.dataset = np.load(self.path_engineered_feature, allow_pickle=True)
        elif fea_used == 'combine':
            self.combine_feature(self.path_learned_feature, self.path_engineered_feature)
            if combine_method == 'normal':
                pass
            if combine_method == 'weight':
                if not '_' in self.algorithm:
                    raise Exception('wrong algorithm')
        
        self.labels = np.load(self.path_labels)
        self.record = pd.read_csv(self.path_record, sep=' ', header=None, names=['index','name','flag'])

        output += 'Total number: {}. Correct number: {}. Incorrect number: {}\n'.format(len(list(self.labels)), list(self.labels).count(1), list(self.labels).count(0))
        print(output, end='')
        self.result += output

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
        output1 += 'Result: \n'

        print(output1, end='')
        self.result += output1

        split_method = self.split_method
        kfold = 10

        # init prediction
        pd = predict_cv.Prediction(self.dataset, self.labels, self.record, self.feature1_length, self.algorithm, split_method, kfold)

        # train and predict
        if split_method == 'cvfold':
            output2 = pd.run_cvfold()
        # train and predict
        elif split_method == 'test_patchsim':
            output2 = pd.run_test_patchsim(self.fea_used)
        elif split_method == 'compare':
            output2 = pd.run_compare()
        elif split_method == 'slice':
            output2 = pd.run_slice()
        else:
            print('wrong split method')
            raise

        self.result += output2

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


parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--w2v', '-w', help='word2vector',)
parser.add_argument('--version', '-v', help='dataset verison', default='V1U')
parser.add_argument('--path', '-p', help='absolute path of dataset', )
parser.add_argument('--task', '-t', help='task', )
args = parser.parse_args()

if __name__ == '__main__':
    # config
    cfg = Config()

    path_dataset = cfg.path_dataset
    path_testdata = cfg.path_testdata
    version = cfg.version
    w2v = cfg.wcv

    # w2v = 'CC2Vec'
    # version = 'V1UCross'

    # w2v = 'CC2Vec'
    # version = 'V1U'

    # w2v = 'Bert'
    # version = 'V2U'

    # w2v = 'CC2Vec'
    # version = 'V2U'

    task = 'experiment'
    # task = args.task

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
        save_feature.save_npy_test(path_dataset, path_testdata, w2v, other, version)

    elif task == 'experiment':
        # start experiment
        print('version: {}  w2c: {}'.format(version, w2v))
        folder = '../data_vector_'+version+'_'+w2v

        path_learned_feature = folder+'/learned_'+w2v+'.npy'
        path_engineered_feature = folder+'/engineered_ods.npy'
        path_labels = folder+'/labels.npy'
        record = folder+'/record.txt'

        # split_method = 'cvfold'
        split_method = 'test_patchsim'
        # split_method = 'compare'
        combine_method = ''

        fea_used = 'learned'
        # fea_used = 'engineered'
        # fea_used = 'combine'

        if fea_used == 'learned':
            # algorithm = 'rf'
            algorithm = 'xgb'
            # algorithm = 'lr'
            # algorithm = 'dnn'
        elif fea_used == 'engineered':
            # algorithm = 'nb'
            # algorithm = 'xgb'
            # algorithm = 'dnn'
            algorithm = 'rf'
        elif fea_used == 'combine':
            # algorithm = 'rf_rf'
            # algorithm = 'xgb_xgb'

            # algorithm = 'lr_combine'
            # algorithm = 'rf'
            algorithm = 'xgb_combine'

            # algorithm = 'dnn_dnn_venn'
            # algorithm = 'wide_deep'


            # combine_method = 'normal'
            # combine_method = 'weight'

        e = Experiment(fea_used, w2v, path_learned_feature, path_engineered_feature, path_testdata, path_labels, record, split_method, algorithm, combine_method )
        e.run()

        
        



