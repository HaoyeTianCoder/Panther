from config_default import *
from common import preprocess, feature_extract, sample
from experiment import predict_cv
from preprocess import deduplicate, obtain_ods_feature, save_feature

class Experiment:
    def __init__(self, fea_used, path_learned_feature, path_engineered_feature, path_labels, split_method, algorithm, combine_method=None):
        self.fea_used = fea_used
        self.combine_method = combine_method
        self.split_method = split_method
        self.algorithm = algorithm
        self.feature1_length = None
        self.result = None

        self.path_learned_feature = path_learned_feature
        self.path_engineered_feature = path_engineered_feature
        self.path_labels = path_labels

        self.dataset = None
        # self.learned_feature = None
        # self.engineered_feature = None
        self.label = None
        # self.embedding_length = embedding_length
        
    def load_combine_data(self, ):
        # load data
        if not os.path.exists(self.path_learned_feature) or not os.path.exists(self.path_engineered_feature) or not os.path.exists(self.path_labels):
            logging.info('path of datset missing ......')
            raise 
        print('------------------------------------')
        print('Loading dataset:')
        
        if fea_used == 'learned':
            self.dataset = np.load(self.path_learned_feature)
        elif fea_used == 'engineered':
            self.dataset = np.load(self.path_engineered_feature)
        elif fea_used == 'combine':
            self.combine_feature(self.path_learned_feature, self.path_engineered_feature)
            if combine_method == 'normal':
                pass
            if combine_method == 'weight':
                self.algorithm = 'lr_rf'
        
        self.labels = np.load(self.path_labels)
        print('Total number: {}. Correct number: {}. Incorrect number: {}'.format(len(list(self.labels)), list(self.labels).count(1), list(self.labels).count(0)))

    def combine_feature(self, path_learned_feature, path_engineered_feature):
        learned_feature = np.load(path_learned_feature)
        engineered_feature = np.load(path_engineered_feature)

        dataset = np.concatenate((learned_feature, engineered_feature),axis=1)

        self.dataset = dataset
        self.feature1_length = learned_feature.shape[1]

    def train_predict(self, ):
        split_method = self.split_method
        # train and predict
        pd = predict_cv.Prediction(self.dataset, self.labels, self.feature1_length, self.algorithm, split_method, 10)

        if split_method == 'cvfold':
            output = pd.run_cvfold()
        elif split_method == 'slice':
            output = pd.run_slice()
        else:
            print('wrong split method')
            raise

        self.result = output

    def run(self, ):
        # load single feature and decide whether combine
        self.load_combine_data()

        # split, train, predict
        print('------------------------------------')
        print('Experiment design: ')
        print('Feature used: {}. Combine_method: {}. ML_algorithm: {}'.format(self.fea_used, self.combine_method, self.algorithm))
        print('------------------------------------')
        print('Result: ')
        self.train_predict()

        # save result
        self.save_result()


    def save_result(self):
        out_path = '../result/'+ self.fea_used + '.result'
        with open(out_path,'w+') as file:
            file.write(self.result)

if __name__ == '__main__':
    # config
    cfg = Config()
    path_dataset = cfg.path_dataset
    dataset_name = cfg.dataset_name
    w2v = cfg.wcv

    task = 'experiment'
    print('TASK: {}'.format(task))

    if task == 'deduplicate':
        # drop same patch
        if 'Unique' in path_dataset:
            print('already deduplicated!')
        else:
            path_dataset, dataset_name = deduplicate.deduplicate_by_token_with_location(dataset_name, path_dataset)

    elif task == 'ods_feature':
        # generate ods feature json under folder where patch is
        obtain_ods_feature.obtain_ods_features(path_dataset)

    elif task == 'save_feature':
        # save learned feature and engineered feature to npy for prediction later
        other = 'ods'
        save_feature.save_features(path_dataset, w2v, other)

    elif task == 'experiment':
        # start experiment
        path_learned_feature = '../data/dataset_learned_Bert.npy'
        path_engineered_feature = '../data/dataset_engineered_ods.npy'
        path_labels = '../data/dataset_labels.npy'

        split_method = 'cvfold'
        algorithm = 'lr'

        # fea_used = 'learned'
        # fea_used = 'engineered'

        # combine
        fea_used = 'combine'
        combine_method = 'normal'
        # combine_method = 'weight'

        e = Experiment(fea_used, path_learned_feature, path_engineered_feature, path_labels, split_method, algorithm, combine_method )
        e.run()

        
        



