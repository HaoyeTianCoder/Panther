from config_default_new import *
from common import preprocess, feature_extract, sample
from experiment import predict_cv
from preprocess import deduplicate, obtain_ods_feature

class Experiment:
    def __init__(self, fea, data_path, label_path, embedding_length, algrithom, w2v=None):
        self.fea = fea
        self.w2v = w2v
        self.algrithom = algrithom
        self.data_path = data_path
        self.label_path = label_path
        self.dataset = None
        self.labels = None
        self.result = None
        self.embedding_length = embedding_length
        
    def load_data(self, ):
        # load data
        if not os.path.exists(self.data_path):
            logging.info('calculating features ......')
            self.get_features()
        print('loading dataset ---------------')
        self.dataset = np.load(self.data_path)
        self.labels = np.load(self.label_path)
        print('total number: {}, correct number: {}, incorrect number: {}'.format(len(list(self.labels)), list(self.labels).count(1), list(self.labels).count(0)))

    def train_predict(self, split_method):
        # train and predict
        pd = predict_cv.Prediction(self.dataset, self.labels, self.embedding_length, self.algrithom, split_method, 10)

        if split_method == 'cvfold':
            output = pd.run_cvfold()
        elif split_method == 'slice':
            output = pd.run_slice()

        return output

    def run(self, split_method):
        print('fea: {}, w2v: {}, data_path: {}, label_path: {}'.format(self.fea, self.w2v, self.data_path, self.label_path))
        self.load_data()
        self.result = self.train_predict(split_method)
        print(self.result)

    def save_result(self):
        out_path = '../result/'+ self.fea + '_' + str(self.w2v) +'.result'
        with open(out_path,'w+') as file:
            file.write(self.result)

if __name__ == '__main__':
    # config
    cfg = Config()
    path_dataset = cfg.path_dataset
    dataset_name = cfg.dataset_name

    task = 'save_feature'
    print('task: {}'.format(task))

    if task == 'deduplicate':
        if 'Unique' in path_dataset:
            print('already deduplicated!')
        else:
            path_dataset, dataset_name = deduplicate.deduplicate_by_token_with_location(dataset_name, path_dataset)
    elif task == 'ods_feature':
        obtain_ods_feature.obtain_ods_features(path_dataset)
    elif task == 'save_feature':
        

        
        



