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

def engineered_features(path_json):
    engineered_vector = []
    try:
        with open(path_json, 'r') as f:
            feature_json = json.load(f)
            features_list = feature_json['files'][0]['features']
            for i in range(len(features_list)):
                dict_fea = features_list[i]
                if 'repairPatterns' in dict_fea.keys():
                    continue
                value = list(dict_fea.values())[0]
                engineered_vector.append(value)
    except Exception as e:
        print('exception: {}, name: {}'.format(e, path_json))
        return []

    return engineered_vector

def save_features(path_dataset, w2v, other):
    total = 0
    cnt = 0
    all_learned_vector = []
    all_engineered_vector = []
    all_label = []
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                total += 1
                name = file.split('.')[0]
                patch = file
                buggy = name + '-s.java'
                fixed = name + '-t.java'
                ods_feature = 'features_' + buggy + '->' + fixed + '.json'

                if '/Correct/' in root:
                    label = 1
                elif '/Incorrect/' in root:
                    label = 0
                else:
                    print('unknow label: {}'.format(name))
                    continue
                all_label.append(label)

                path_patch = os.path.join(root, patch)

                if w2v:
                    learned_vector = learned_feature(path_patch, w2v)
                if other:
                    if other == 'ods':
                        path_json = os.path.join(root, ods_feature)
                        engineered_vector = engineered_features(path_json)
                    else:
                        raise

                if learned_vector == [] or engineered_vector == []:
                    continue
                all_learned_vector.append(learned_vector)
                all_engineered_vector.append(engineered_vector)

                cnt += 1
                print('process {}/{}, patch name: {}'.format(cnt, total, patch))

    # save
    path_learned_features = '../data/' + 'dataset_learned_' + w2v + '.npy'
    path_engineered_features = '../data/' + 'dataset_engineered_' + other + '.npy'
    path_labels = '../data/dataset_labels.npy'
    np.save(path_learned_features, all_learned_vector)
    np.save(path_engineered_features, all_engineered_vector)
    np.save(path_labels, all_label)

def learned_feature(path_patch, w2v):
    try:
        bugy_all = get_diff_files_frag(path_patch, type='patched')
        patched_all = get_diff_files_frag(path_patch, type='buggy')
    except Exception as e:
        print('exception: {}, name: {}'.format(e,path_patch))
        return []

    # tokenize word
    bugy_all_token = word_tokenize(bugy_all)
    patched_all_token = word_tokenize(patched_all)

    bug_vec, patched_vec = output_vec(w2v, bugy_all_token, patched_all_token)

    bug_vec = bug_vec.reshape((1, -1))
    patched_vec = patched_vec.reshape((1, -1))

    # embedding feature cross
    subtract, multiple, cos, euc = multi_diff_features(bug_vec, patched_vec)
    # embedding = subtract + multiple + [cos] + [euc]
    embedding = np.hstack((subtract, multiple, cos, euc,))

    return list(embedding.flatten())

def subtraction(buggy, patched):
    return buggy - patched

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
        m = BertClient(check_length=False)
        bug_vec = m.encode([bugy_all_token], is_tokenized=True)
        patched_vec = m.encode([patched_all_token], is_tokenized=True)
    elif w2v == 'Doc':
        m = Doc2Vec.load('/Users/haoye.tian/Documents/University/project/patch_predict/data/model/doc_frag.model')
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
