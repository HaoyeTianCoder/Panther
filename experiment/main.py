from config_default import *
from process import preprocess, feature_extract, sample
from experiment import predict_cv

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

    def get_features(self, ):
        # w2v = 'bert'
        # w2v = 'doc'

        s = sample.Sample()
        f = feature_extract.Feature(self.fea, self.w2v)

        dataset_correct, labels_correct, all_buggy_patched_correct, correct_nubmer = f.feature_obtain_correct_sample(
            correct_patches, correct_engineerings, )
        s.dataset = dataset_correct
        s.labels = labels_correct
        s.buggy_patched = all_buggy_patched_correct

        # add dataset
        dataset_incorrect, labels_incorrcet, all_buggy_patched_incorrect, incorrect_number = f.feature_obtain_incorrect_sample(
            incorrect_patches, incorrect_engineerings)
        s.add_data(dataset_incorrect, labels_incorrcet, all_buggy_patched_incorrect)

        # count
        s.correct_nubmer = correct_nubmer
        s.incorrect_number = incorrect_number
        s.total_number = s.correct_nubmer + s.incorrect_number
        print('total number: {}, correct number: {}. incorrect number: {}'.format(s.total_number, s.correct_nubmer, s.incorrect_number))

        # deduplicate
        print('Deduplicating ------------------')
        s.deduplicate()

        print('total number: {}, correct number: {}, incorrect number: {}'.format(s.total_number, s.correct_nubmer, s.incorrect_number))

        # save data and label
        np.save(self.data_path, s.dataset)
        np.save(self.label_path, s.labels)
        
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
    correct_patches = cfg.correct_patches
    correct_engineerings = cfg.correct_engineering_features
    incorrect_patches = cfg.incorrect_patches
    incorrect_engineerings = cfg.incorrect_engineering_features

    # correct_patches = ['/Users/haoye.tian/Documents/University/data/Exp-2-data-deduplicate/correct-patches']
    # incorrect_patches = ['/Users/haoye.tian/Documents/University/data/Exp-2-data-deduplicate/incorrect-patches']

    exp_dict = {'embeddings': 'lr', 'engineerings': 'rf', 'combinings': 'lr_rf'}
    feas = ['embeddings', 'engineerings', 'combinings']

    # hyper-parameter
    fea = feas[2]
    split_method = 'cvfold'
    # split_method = 'slice'
    algrithom = exp_dict[fea]
    embedding_length = -2050

    data_path, label_path, w2v = None, None, None
    if fea == 'embeddings':
        w2v = 'bert'
        data_path = '../data/dataset_'+ w2v +'.npy'
        label_path = '../data/labels_'+ w2v +'.npy'
    elif fea == 'engineerings':
        data_path = '../data/dataset_engineerings.npy'
        label_path = '../data/labels_engineerings.npy'
    elif fea == 'combinings':
        w2v = 'bert'
        data_path = '../data/dataset_combinings_'+ w2v +'.npy'
        label_path = '../data/labels_combinings_'+ w2v +'.npy'
    else:
        print('wrong type...')

    # data_path = '../data/dataset_embeddings_tmp.npy'
    # label_path = '../data/labels_embeddings_tmp.npy'

    # start experiment
    e = Experiment(fea, data_path, label_path, embedding_length, algrithom, w2v,)
    e.run(split_method)
    e.save_result()
        
        



