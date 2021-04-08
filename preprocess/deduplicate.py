import os
import shutil
from subprocess import *
from nltk.tokenize import word_tokenize

def prepare_legal_file(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]

                new_path = root.replace('PatchCollecting','PatchCollectingV2')
                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                old_patch = new_patch = file

                old_buggy = name + '.buggy'
                old_fixed = name + '.fixed'

                new_buggy = name + '_s.java'
                new_fixed = name + '_t.java'

                shutil.copy(os.path.join(root, old_patch), os.path.join(new_path, new_patch))
                shutil.copy(os.path.join(root, old_buggy), os.path.join(new_path, new_buggy))
                shutil.copy(os.path.join(root, old_fixed), os.path.join(new_path, new_fixed))

def deduplicate_by_content_with_location(dataset_name, path_dataset):
    new_dataset_name = dataset_name + 'UniqueNormal'
    new_dataset_path = path_dataset.replace(dataset_name, new_dataset_name)
    unique_dict = {}
    pre = exception = post = repeat = 0
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                # unlabeled
                if 'iFixR' in root:
                    continue
                path_patch = os.path.join(root, file)
                unique_str = ''
                pre += 1
                print('{}, name: {}'.format(pre, file.split('.')[0]))
                try:
                    with open(path_patch, 'r') as f:
                        foundAT = False
                        for line in f:
                            if line.startswith('--') or line.startswith('++'):
                                continue
                            if not foundAT and not line.startswith('@@ '):
                                continue
                            else:
                                if line.startswith('@@ '):
                                    foundAT = True
                                    unique_str += line.strip() + ' '
                                elif line.startswith('-') or line.startswith('+'):
                                    unique_str += line[1:].strip() + ' '
                                else:
                                    unique_str += line.strip() + ' '
                except Exception as e:
                    print('Exception: {}'.format(e))
                    exception += 1
                    continue

                if unique_str in unique_dict:
                    unique_dict[unique_str] += 1
                    repeat += 1
                    continue
                else:
                    unique_dict[unique_str] = 0

                    # copy unique to another folder
                    name = file.split('.')[0]
                    new_path = root.replace(dataset_name, new_dataset_name)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    patch = file
                    buggy = name + '_s.java'
                    fixed = name + '_t.java'
                    feature = 'features_' + buggy + '->' + fixed + '.json'

                    shutil.copy(os.path.join(root, patch), os.path.join(new_path, patch))
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_path, buggy))
                    shutil.copy(os.path.join(root, fixed), os.path.join(new_path, fixed))
                    if os.path.exists(os.path.join(root, feature)):
                        shutil.copy(os.path.join(root, feature), os.path.join(new_path, feature))

                    post += 1

    print('pre:{}, post:{} ---- exception:{}, repeat:{}'.format(pre, post, exception, repeat))
    return new_dataset_path, new_dataset_name


def deduplicate_by_token_with_location(dataset_name, path_dataset):
    new_dataset_name = dataset_name + 'Unique'
    new_dataset_path = path_dataset.replace(dataset_name, new_dataset_name)
    unique_dict = {}
    pre = exception = post = repeat = 0
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                # unlabeled
                if 'iFixR' in root:
                    continue
                path_patch = os.path.join(root, file)
                unique_str = ''
                pre += 1
                print('{}, name: {}'.format(pre, file.split('.')[0]))
                try:
                    with open(path_patch, 'r') as f:
                        foundAT = False
                        for line in f:
                            if line.startswith('--') or line.startswith('++'):
                                continue
                            if not foundAT and not line.startswith('@@ '):
                                continue
                            else:
                                if line.startswith('@@ '):
                                    foundAT = True
                                    unique_str += ' '.join(word_tokenize(line.strip())) + ' '
                                elif line.startswith('-') or line.startswith('+'):
                                    unique_str += ' '.join(word_tokenize(line[1:].strip())) + ' '
                                else:
                                    unique_str += ' '.join(word_tokenize(line.strip())) + ' '
                except Exception as e:
                    print('Exception: {}'.format(e))
                    exception += 1
                    continue

                if unique_str in unique_dict:
                    unique_dict[unique_str] += 1
                    repeat += 1
                    continue
                else:
                    unique_dict[unique_str] = 0

                    # copy unique to another folder
                    name = file.split('.')[0]
                    new_path = root.replace(dataset_name, new_dataset_name)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    patch = file
                    buggy = name + '_s.java'
                    fixed = name + '_t.java'
                    feature = 'features_' + buggy + '->' + fixed + '.json'

                    try:
                        shutil.copy(os.path.join(root, patch), os.path.join(new_path, patch))
                        shutil.copy(os.path.join(root, buggy), os.path.join(new_path, buggy))
                        shutil.copy(os.path.join(root, fixed), os.path.join(new_path, fixed))
                        if os.path.exists(os.path.join(root, feature)):
                            shutil.copy(os.path.join(root, feature), os.path.join(new_path, feature))
                    except Exception as e:
                        print(e)
                        continue

                    post += 1

    print('pre:{}, post:{} ---- exception:{}, repeat:{}'.format(pre, post, exception, repeat))
    print('remember change path in config_default.py !!!')
    return new_dataset_path, new_dataset_name

if __name__ == '__main__':
    # path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'
    # prepare_legal_file(path)

    # v2 contains ods feature
    dataset_name = 'PatchCollectingV2'
    path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2/'

    # by content with location number
    # deduplicate_by_content_with_location(path2)

    # by token content with location number
    deduplicate_by_token_with_location(dataset_name, path_dataset)