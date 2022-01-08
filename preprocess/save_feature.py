import os
import shutil
from subprocess import *
import numpy as np
from bert_serving.client import BertClient
from gensim.models import word2vec,Doc2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import *
import re
import json
import pickle
from CC2Vec import lmg_cc2ftr_interface

def engineered_features(path_json):
    other_vector = []
    P4J_vector = []
    repair_patterns = []
    repair_patterns2 = []
    try:
        with open(path_json, 'r') as f:
            feature_json = json.load(f)
            features_list = feature_json['files'][0]['features']
            P4J = features_list[-3]
            RP = features_list[-2]
            RP2 = features_list[-1]

            '''
            # other
            for k,v in other.items():
                # if k.startswith('FEATURES_BINARYOPERATOR'):
                #     for k2,v2 in other[k].items():
                #         for k3,v3 in other[k][k2].items():
                #             if v3 == 'true':
                #                 other_vector.append('1')
                #             elif v3 == 'false':
                #                 other_vector.append('0')
                #             else:
                #                 other_vector.append('0.5')
                if k.startswith('S'):
                    if k.startswith('S6'):
                        continue
                    other_vector.append(v)
                else:
                    continue
            '''

            # P4J
            if not list(P4J.keys())[100].startswith('P4J'):
                raise
            for k,v in P4J.items():
                # dict = P4J[i]
                # value = list(dict.values())[0]
                P4J_vector.append(int(v))

            # repair pattern
            for k,v in RP['repairPatterns'].items():
                repair_patterns.append(v)

            # repair pattern 2
            for k,v in RP2.items():
                repair_patterns2.append(v)

            # for i in range(len(features_list)):
            #     dict_fea = features_list[i]
            #     if 'repairPatterns' in dict_fea.keys():
            #             # continue
            #             for k,v in dict_fea['repairPatterns'].items():
            #                 repair_patterns.append(int(v))
            #     else:
            #         value = list(dict_fea.values())[0]
            #         engineered_vector.append(value)
    except Exception as e:
        print('name: {}, exception: {}'.format(path_json, e))
        return []

    if len(P4J_vector) != 156 or len(repair_patterns) != 26 or len(repair_patterns2) != 13:
        print('name: {}, exception: {}'.format(path_json, 'null feature or shape error'))
        return []

    # return engineered_vector
    return P4J_vector + repair_patterns + repair_patterns2

def save_npy(path_dataset, w2v, other, version):
    total = -1
    cnt = 0
    all_learned_vector = []
    all_engineered_vector = []
    all_label = []
    record = ''
    dictionary = pickle.load(open('../CC2Vec/dict.pkl', 'rb'))
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                total += 1
                name = file.split('.')[0]
                patch = file
                buggy = name + '-s.java'
                fixed = name + '-t.java'
                ods_feature = 'features_' + name.split('_')[0] + '.json'

                if '/Correct/' in root:
                    label = 1
                elif '/Incorrect/' in root:
                    label = 0
                else:
                    print('unknow label: {}'.format(name))
                    raise

                path_patch = os.path.join(root, patch)

                # engineered feature
                if other == 'ods':
                    path_json = os.path.join(root, ods_feature)
                    engineered_vector = engineered_features(path_json)
                else:
                    raise
                if engineered_vector == []:
                    continue

                # learned feature
                if w2v == 'CC2Vec':
                    learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch, load_model='../CC2Vec/cc2ftr.pt', dictionary=dictionary)
                    learned_vector = list(learned_vector[0])
                elif w2v == 'Bert' or w2v == 'Doc':
                    learned_vector, patch_frag = learned_feature(path_patch, w2v)
                else:
                    raise
                # learned_vector = [1,2,3,4]
                if learned_vector == []:
                    continue

                all_learned_vector.append(learned_vector)
                all_engineered_vector.append(engineered_vector)
                all_label.append(label)

                print('process {}/{}, patch name: {}'.format(cnt, total, patch))
                # f.write('{} {} {}\n'.format(cnt, name, 'success'))
                record += '{} {} {}\n'.format(cnt, str(label)+'-'+name, 'success')

                cnt += 1
                if len(learned_vector) != len(all_learned_vector[0]) or len(engineered_vector) != len(all_engineered_vector[0]):
                    raise Exception('shape error')
    # save
    folder = '../data_vector_' + version + '_' + w2v
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_learned_features = folder + '/learned_' + w2v + '.npy'
    path_engineered_features = folder + '/engineered_' + other + '.npy'
    path_labels = folder + '/labels.npy'

    f = open(folder + '/record.txt', 'w+')
    f.write(record)
    f.close()
    np.save(path_learned_features, all_learned_vector, allow_pickle=False)
    np.save(path_engineered_features, all_engineered_vector, allow_pickle=False)
    np.save(path_labels, all_label, allow_pickle=False)

def save_npy_test(path_dataset, path_testdata, w2v, other, version):
    total = -1
    cnt = 0
    all_learned_vector = []
    all_engineered_vector = []
    all_label = []
    record = ''
    dictionary = pickle.load(open('../CC2Vec/dict.pkl', 'rb'))

    # coding...
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                total += 1
                name = file.split('.')[0]
                patch = file
                buggy = name + '-s.java'
                fixed = name + '-t.java'
                ods_feature = 'features_' + name.split('_')[0] + '.json'

                if '/Correct/' in root:
                    label = 1
                elif '/Incorrect/' in root:
                    label = 0
                else:
                    print('unknow label: {}'.format(name))
                    raise

                path_patch = os.path.join(root, patch)

                # engineered feature
                if other == 'ods':
                    path_json = os.path.join(root, ods_feature)
                    engineered_vector = engineered_features(path_json)
                else:
                    raise
                if engineered_vector == []:
                    continue

                # learned feature
                if w2v == 'CC2Vec':
                    learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch, load_model='../CC2Vec/cc2ftr.pt', dictionary=dictionary)
                    learned_vector = list(learned_vector[0])
                elif w2v == 'Bert' or w2v == 'Doc':
                    learned_vector, patch_frag = learned_feature(path_patch, w2v)
                else:
                    raise
                # learned_vector = [1,2,3,4]
                if learned_vector == []:
                    continue

                all_learned_vector.append(learned_vector)
                all_engineered_vector.append(engineered_vector)
                all_label.append(label)

                print('process {}/{}, patch name: {}'.format(cnt, total, patch))
                # f.write('{} {} {}\n'.format(cnt, name, 'success'))
                record += '{} {} {}\n'.format(cnt, str(label)+'-'+name, 'success')

                cnt += 1
                if len(learned_vector) != len(all_learned_vector[0]) or len(engineered_vector) != len(all_engineered_vector[0]):
                    raise Exception('shape error')
    # save
    folder = '../data_vector_' + version + '_' + w2v
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_learned_features = folder + '/learned_' + w2v + '.npy'
    path_engineered_features = folder + '/engineered_' + other + '.npy'
    path_labels = folder + '/labels.npy'

    f = open(folder + '/record.txt', 'w+')
    f.write(record)
    f.close()
    np.save(path_learned_features, all_learned_vector, allow_pickle=False)
    np.save(path_engineered_features, all_engineered_vector, allow_pickle=False)
    np.save(path_labels, all_label, allow_pickle=False)

def learned_feature(path_patch, w2v):
    try:
        bugy_all = get_diff_files_frag(path_patch, type='buggy')
        patched_all = get_diff_files_frag(path_patch, type='patched')
    except Exception as e:
        print('name: {}, exception: {}'.format(path_patch, e))
        return []

    # tokenize word
    bugy_all_token = word_tokenize(bugy_all)
    patched_all_token = word_tokenize(patched_all)

    try:
        bug_vec, patched_vec = output_vec(w2v, bugy_all_token, patched_all_token)
    except Exception as e:
        print('name: {}, exception: {}'.format(path_patch, e))
        return []

    bug_vec = bug_vec.reshape((1, -1))
    patched_vec = patched_vec.reshape((1, -1))

    # embedding feature cross
    subtract, multiple, cos, euc = multi_diff_features(bug_vec, patched_vec)
    embedding = np.hstack((subtract, multiple, cos, euc,))

    # embedding = subtraction(bug_vec, patched_vec)

    return list(embedding.flatten()), bugy_all+patched_all

def subtraction(buggy, patched):
    return patched - buggy

def multiplication(buggy, patched):
    return buggy * patched

def cosine_similarity(buggy, patched):
    return paired_cosine_distances(buggy, patched)

def euclidean_similarity(buggy, patched):
    return paired_euclidean_distances(buggy, patched)

def multi_diff_features(buggy, patched):
    subtract = subtraction(buggy, patched)
    multiple = multiplication(buggy, patched)
    cos = cosine_similarity(buggy, patched).reshape((1, 1))
    euc = euclidean_similarity(buggy, patched).reshape((1, 1))

    return subtract, multiple, cos, euc

def output_vec(w2v, bugy_all_token, patched_all_token):

    if w2v == 'Bert':
        m = BertClient(check_length=False, check_version=False)
        bug_vec = m.encode([bugy_all_token], is_tokenized=True)
        patched_vec = m.encode([patched_all_token], is_tokenized=True)
    elif w2v == 'Doc':
        # m = Doc2Vec.load('../model/doc_file_64d.model')
        m = Doc2Vec.load('../model/Doc_frag_ASE.model')
        bug_vec = m.infer_vector(bugy_all_token, alpha=0.025, steps=300)
        patched_vec = m.infer_vector(patched_all_token, alpha=0.025, steps=300)
    else:
        print('wrong model')
        raise

    return bug_vec, patched_vec

def get_diff_files_frag( path_patch, type):
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
                        # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        # lines += ' '.join(line) + ' '
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
                        # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        # lines += ' '.join(line) + ' '
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

if __name__ == '__main__':
    path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2UniqueToken'
