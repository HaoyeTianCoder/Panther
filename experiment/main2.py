from config_default import *
from common import preprocess, feature_extract, sample
from experiment import predict_cv, train_doc
from preprocess import deduplicate, obtain_ods_feature, save_feature
import pandas as pd

class Experiment:
    def __init__(self, fea_used, path_learned_feature, path_engineered_feature, path_labels, record, split_method, algorithm, combine_method=None):
        self.fea_used = fea_used
        self.combine_method = combine_method
        self.split_method = split_method
        self.algorithm = algorithm
        self.feature1_length = None
        self.result = ''

        self.path_learned_feature = path_learned_feature
        self.path_engineered_feature = path_engineered_feature
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
            raise Exception('miss the path of the datset ......')
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

        dataset = np.concatenate((learned_feature, engineered_feature),axis=1)

        self.dataset = dataset
        self.feature1_length = learned_feature.shape[1]

    def train_predict(self, ):
        output1 = '------------------------------------\n'
        output1 += 'Experiment design: \n'
        output1 += 'Feature used: {}. Combine_method: {}. ML_algorithm: {}\n'.format(self.fea_used, self.combine_method, self.algorithm)
        output1 += '------------------------------------\n'
        output1 += 'Result: \n'

        print(output1, end='')
        self.result += output1

        split_method = self.split_method
        kfold = 10

        # init prediction
        pd = predict_cv.Prediction(self.dataset, self.labels, self.record,self.feature1_length, self.algorithm, split_method, kfold)

        # train and predict
        if split_method == 'cvfold':
            output2 = pd.run_cvfold()
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

if __name__ == '__main__':
    # config
    cfg = Config()
    path_dataset = cfg.path_dataset
    dataset_name = cfg.dataset_name
    version = cfg.version
    w2v = cfg.wcv
    # w2v = 'Doc'

    path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2UniqueToken'
    dataset_name = 'PatchCollectingV2UniqueToken'
    version = 'V2U'

    task = 'save_npy'
    print('TASK: {}'.format(task))

    if task == 'deduplicate':
        # drop same patch
        if 'Unique' in path_dataset:
            print('already deduplicated!')
        else:
            path_dataset, dataset_name = deduplicate.deduplicate_by_token_with_location(dataset_name, path_dataset)

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
        save_feature.save_npy(path_dataset, w2v, other, version)

    elif task == 'experiment':
        # start experiment
        path_learned_feature = '../data_vector_Fix_Apart_Bert/learned_Bert.npy'
        path_engineered_feature = '../data_vector_Fix_Apart_Bert/engineered_ods.npy'
        path_labels = '../data_vector_Fix_Apart_Bert/labels.npy'
        record = '../data_vector_Fix_Apart_Bert/record.txt'

        split_method = 'cvfold'
        combine_method = ''

        # fea_used = 'learned'
        # fea_used = 'engineered'
        fea_used = 'combine'

        if fea_used == 'learned':
            algorithm = 'xgb'
            # algorithm = 'lr'
        elif fea_used == 'engineered':
            algorithm = 'xgb'
            # algorithm = 'lr'
        elif fea_used == 'combine':
            # algorithm = 'xgb_venn'
            # algorithm = 'lr_xgb_dnn'
            # algorithm = 'lr_xgb'
            algorithm = 'xgb_xgb'

            # combine_method = 'normal'
            # combine_method = 'weight'

        e = Experiment(fea_used, path_learned_feature, path_engineered_feature, path_labels, record, split_method, algorithm, combine_method )
        e.run()

        
        



