from config_default import *
from process import preprocess, engineering_feature, sample
from experiment import predict_cv

if __name__ == '__main__':
    cfg = Config()
    correct_patches = cfg.correct_patches
    correct_engineerings = cfg.correct_engineering_features
    incorrect_patches = cfg.incorrect_patches
    incorrect_engineerings = cfg.incorrect_engineering_features

    w2v = 'bert'
    # w2v = 'doc'

    s = sample.Sample()
    f = engineering_feature.Feature(w2v)

    dataset_correct, labels_correct, all_buggy_patched_correct, correct_nubmer = f.feature_obtain_correct_sample(correct_patches, correct_engineerings, )
    s.dataset = dataset_correct
    s.labels = labels_correct
    s.buggy_patched = all_buggy_patched_correct

    # add dataset
    dataset_incorrect, labels_incorrcet, all_buggy_patched_incorrect, incorrect_number = f.feature_obtain_incorrect_sample(incorrect_patches, incorrect_engineerings)
    s.add_data(dataset_incorrect, labels_incorrcet, all_buggy_patched_incorrect)

    # count
    s.correct_nubmer = correct_nubmer
    s.incorrect_number = incorrect_number
    s.total_number = s.correct_nubmer + s.incorrect_number

    print('total number: {}'.format(s.total_number))
    print('correct number: {}. incorrect number: {}'.format(s.correct_nubmer, s.incorrect_number))

    # deduplicate
    s.deduplicate()

    print('deduplicat ------------------')
    print('total number: {}'.format(s.total_number))
    print('correct number: {}. incorrect number: {}'.format(s.correct_nubmer, s.incorrect_number))

    # train and predict
    algorithm = 'rf'
    pd = predict_cv.Prediction(s.dataset, s.labels, algorithm, 10)
    pd.run()
