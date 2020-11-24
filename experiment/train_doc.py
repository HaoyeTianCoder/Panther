import os
from preprocess import save_feature
from nltk.tokenize import word_tokenize
from gensim.models import word2vec, Doc2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class doc:
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

    def get_doc_content(self, path):
        with open(path, 'r') as f:
            lines = ''
            for line in f:
                line = line.strip('\t').strip()
                if not (line.startswith('/') or line.startswith('*') or line.startswith('package') or line.startswith('import')):
                    lines += line + ' '
            return lines

    def train(self):
        all_buggy = []
        all_patched = []
        cnt = 0
        for root, dirs, files in os.walk(self.path_dataset):
            for file in files:
                if file.endswith('.patch'):
                    name = file.split('.')[0]
                    patch = file
                    buggy = name + '-s.java'
                    fixed = name + '-t.java'
                    path_patch = os.path.join(root, patch)
                    path_buggy = os.path.join(root, buggy)
                    path_patched = os.path.join(root,fixed)

                    try:
                        # bugy_all = save_feature.get_diff_files_frag(path_patch, type='patched')
                        # patched_all = save_feature.get_diff_files_frag(path_patch, type='buggy')
                        bugy_all = self.get_doc_content(path_buggy)
                        patched_all = self.get_doc_content(path_patched)
                    except Exception as e:
                        print('name: {}, exception: {}'.format(path_patch, e))
                        continue

                    # tokenize word
                    bugy_all_token = word_tokenize(bugy_all)
                    patched_all_token = word_tokenize(patched_all)

                    all_buggy.append(bugy_all_token)
                    all_patched.append(patched_all_token)

                    cnt += 1
                    print('{}: name: {}'.format(cnt, name))
        data = all_buggy + all_patched
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
        model = Doc2Vec(documents, vector_size=64, window=5, min_count=1, workers=4)
        model.save('../model/doc_file_64d.model')