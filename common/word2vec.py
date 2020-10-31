from bert_serving.client import BertClient
from gensim.models import word2vec,Doc2Vec

class W2v:
    def __init__(self, w2vName):
        self.w2v = None
        if w2vName == 'bert':
            self.w2v = self.Bert1()
        elif w2vName == 'doc':
            self.w2v = self.Doc1()
        elif w2vName == None:
            self.w2v = None
        else:
            raise NameError('wrong name')

    class Bert1:
        def __init__(self):
            # max_seq_len=360, output = 1024
            self.m = BertClient(check_length=False)
        def obtain(self):
            return self.m
        def output_vec(self, bugy_all, patched_all):
            bug_vec = self.m.encode([bugy_all], is_tokenized=True)
            patched_vec = self.m.encode([patched_all], is_tokenized=True)

            return bug_vec, patched_vec

    class Doc1:
        def __init__(self):
            self.m = Doc2Vec.load('/Users/haoye.tian/Documents/University/project/patch_predict/data/model/doc_frag.model')
        def obtain(self):
            return self.m
        def output_vec(self, bugy_all, patched_all):
            bug_vec = self.m.infer_vector(bugy_all, alpha=0.025, steps=300)
            patched_vec = self.m.infer_vector(patched_all, alpha=0.025, steps=300)
            # similarity calculation
            # result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
            return bug_vec, patched_vec
