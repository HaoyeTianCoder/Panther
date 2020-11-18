import os
from config_default_old import *
from bert_serving.client import BertClient
import json
import numpy as np
# m = BertClient(check_length=False)
from nltk.tokenize import word_tokenize
from common import word2vec
from sklearn.metrics.pairwise import *
import re


class Feature:
    def __init__(self, fea, w2vName):
        self.logger = logging.getLogger(__name__)
        self.sample_number = 0
        self.error_number = 0
        self.fea = fea
        self.w2v_entry = word2vec.W2v(w2vName).w2v


    def get_diff_files_frag(self, path_patch, type):
        with open(path_patch, 'r') as file:
            lines = ''
            p = r"([^\w_])"
            flag = True
            # try:
            for line in file:
                line = line.strip()
                if '*/' in line:
                    flag = True
                    continue
                if flag == False:
                    continue
                if line != '':
                    if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                        continue
                    if line.startswith('Index') or line.startswith('==='):
                        continue
                    elif '/*' in line:
                        flag = False
                        continue
                    elif type == 'buggy':
                        if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                            continue
                            line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            lines += ' '.join(line) + ' '
                        elif line.startswith('-'):
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            line = [x.strip() for x in line]
                            while '' in line:
                                line.remove('')
                            line = ' '.join(line)
                            lines += line.strip() + ' '
                        elif line.startswith('+'):
                            # do nothing
                            pass
                        else:
                            line = re.split(pattern=p, string=line.strip())
                            line = [x.strip() for x in line]
                            while '' in line:
                                line.remove('')
                            line = ' '.join(line)
                            lines += line.strip() + ' '
                    elif type == 'patched':
                        if line.startswith('+++'):
                            continue
                            line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            lines += ' '.join(line) + ' '
                        elif line.startswith('+'):
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            line = [x.strip() for x in line]
                            while '' in line:
                                line.remove('')
                            line = ' '.join(line)
                            lines += line.strip() + ' '
                        elif line.startswith('-'):
                            # do nothing
                            pass
                        else:
                            line = re.split(pattern=p, string=line.strip())
                            line = [x.strip() for x in line]
                            while '' in line:
                                line.remove('')
                            line = ' '.join(line)
                            lines += line.strip() + ' '
            # except Exception:
            #     print(Exception)
            #     return 'Error'
            return lines

    def subtraction(self, buggy, patched):
        return buggy - patched

    def multiplication(self, buggy, patched):
        return buggy * patched

    def cosine_similarity(self, buggy, patched):
        return paired_cosine_distances(buggy, patched)

    def euclidean_similarity(self, buggy, patched):
        return paired_euclidean_distances(buggy, patched)

    def embedding_diff_features(self, buggy, patched):

        subtract = self.subtraction(buggy, patched)
        multiple = self.multiplication(buggy, patched)
        cos = self.cosine_similarity(buggy, patched).reshape((1, 1))
        euc = self.euclidean_similarity(buggy, patched).reshape((1, 1))

        return subtract, multiple, cos, euc

    def embedding_feature(self, path_patch, ):
        try:
            bugy_all = self.get_diff_files_frag(path_patch, type='patched')
            patched_all = self.get_diff_files_frag(path_patch, type='buggy')
        except Exception as e:
            self.logger.debug(e)
            raise

        # tokenize word
        bugy_all2 = word_tokenize(bugy_all)
        patched_all2 = word_tokenize(patched_all)

        # if w2v == 'bert':
        #     bug_vec, patched_vec = self.w2v_entry.output_vec(bugy_all2, patched_all2)
        # elif w2v == 'doc':

        if self.fea == 'engineerings' and self.w2v_entry == None:
            return [], bugy_all+patched_all

        bug_vec, patched_vec = self.w2v_entry.output_vec(bugy_all2, patched_all2)


        bug_vec = bug_vec.reshape((1, -1))
        patched_vec = patched_vec.reshape((1, -1))

        # embedding feature cross
        subtract, multiple, cos, euc = self.embedding_diff_features(bug_vec, patched_vec)
        # embedding = subtract + multiple + [cos] + [euc]
        embedding = np.hstack((subtract, multiple, cos, euc,))

        return list(embedding.flatten()), bugy_all+patched_all

    def feature_obtain_incorrect_sample(self, patches, engineerings):
        # label = 0.0
        incorrect_number = 0
        all_vector = []
        all_buggy_patched = []
        for i in range(len(patches)):
            patch = patches[i]
            engineering = engineerings[i]
            for root, dirs, files in os.walk(patch):
                for file in files:
                    subPath = root.replace(patch, '')
                    bugName = subPath.split('/')[1]
                    toolName = subPath.split('/')[2]

                    if toolName.startswith('Arja') or toolName.startswith('GenProg'):
                        continue
                    if not file.endswith('.patch'):
                        continue
                    # embedding = [np.random.random_sample()]
                    try:
                        embedding, buggy_patched = self.embedding_feature(os.path.join(root, file),)
                    except Exception as e:
                        self.logger.error(e)
                        continue

                    # if self.fea != 'embeddings':

                    if bugName.startswith('Closure'):
                        engineeringName = 'P4Jfeatures_' + bugName + '-' + toolName + '.json'
                    else:
                        engineeringName = 'P4Jfeatures_' + bugName + '.json'
                    engineering_path = os.path.join(engineering, engineeringName)
                    try:
                        with open(engineering_path, 'r+') as jfile:
                            dict = json.load(jfile)
                            engineering_features_dict = dict['files'][0]['features'][0]
                            # sort and obtain features
                            engineering_features_dict_sorted = sorted(engineering_features_dict.items(), key=lambda d: d[0])
                            engineering_features_list = [float(value) for key, value in engineering_features_dict_sorted]
                            if engineering_features_list == []:
                                continue
                    except Exception as e:
                        # print('********: {}'.format(engineering_path))
                        self.error_number += 1

                        # keep the same number of dataset with embeddings
                        continue

                    # separate or combine
                    if self.fea == 'embeddings':
                        vector = embedding
                    elif self.fea == 'engineerings':
                        vector = engineering_features_list
                    elif self.fea == 'combinings':
                        vector = engineering_features_list + embedding

                    all_buggy_patched.append([buggy_patched])
                    all_vector.append(vector)
                    incorrect_number += 1
                    self.sample_number += 1
                    self.logger.info('sample number: {}'.format(self.sample_number))

        self.logger.info('error number: {}'.format(self.error_number))
        return np.array(all_vector), np.zeros(len(all_vector)), np.array(all_buggy_patched), incorrect_number

    def feature_obtain_correct_sample(self, patches, engineerings,):
        # label = 1.0
        correct_number = 0
        all_vector = []
        all_buggy_patched = []
        for i in range(len(patches)):
            patch = patches[i]
            engineering = engineerings[i]
            for root, dirs, files in os.walk(patch):
                for file in files:
                    if not file.endswith('.patch'):
                        continue
                    # embedding = [np.random.random_sample()]
                    try:
                        embedding, buggy_patched = self.embedding_feature(os.path.join(root, file),)
                    except Exception as e:
                        self.logger.error(e)
                        continue

                    # if self.fea != 'embeddings':
                    subPath = root.replace(patch, '')
                    bugName = subPath.split('/')[1]
                    engineeringName = 'P4Jfeatures_' + bugName + '.json'
                    engineering_path = os.path.join(engineering, engineeringName)

                    try:
                        with open(engineering_path, 'r+') as jfile:
                            dict = json.load(jfile)
                            engineering_features_dict = dict['files'][0]['features'][0]
                            # sort and obtain features
                            engineering_features_dict_sorted = sorted(engineering_features_dict.items(), key=lambda d: d[0])
                            engineering_features_list = [float(value) for key, value in engineering_features_dict_sorted]
                            if engineering_features_list == []:
                                continue
                    except Exception as e:
                        self.error_number += 1

                        # keep the same number of dataset with embeddings
                        continue

                    # separate or combine
                    if self.fea == 'embeddings':
                        vector = embedding
                    elif self.fea == 'engineerings':
                        vector = engineering_features_list
                    elif self.fea == 'combinings':
                        vector = engineering_features_list + embedding

                    all_vector.append(vector)
                    all_buggy_patched.append([buggy_patched])
                    correct_number += 1
                    self.sample_number += 1
                    self.logger.info('sample number: {}'.format(self.sample_number))

        self.logger.info('error number: {}'.format(self.error_number))
        return np.array(all_vector), np.ones(len(all_vector)), np.array(all_buggy_patched), correct_number



if __name__ == '__main__':
    cfg = Config()
    correct_patches = cfg.correct_patches
    correct_engineerings = cfg.correct_engineering_features
    incorrect_patches = cfg.incorrect_patches
    incorrect_engineerings = cfg.incorrect_engineering_features

    w2v = 'bert'
    fea = 'engineerings'
    f = Feature(fea, w2v)

    # init
    dataset_correct, labels_correct, buggy_patched_correct, correct_nubmer = f.feature_obtain_correct_sample(correct_patches, correct_engineerings, )

    # add dataset feature
    dataset_incorrect, labels_incorrcet, buggy_patched_incorrect, incorrect_number = f.feature_obtain_incorrect_sample(incorrect_patches, incorrect_engineerings)


