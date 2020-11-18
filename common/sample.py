from config_default_old import *

class Sample:
    def __init__(self):
        self.dataset = None
        self.correct_nubmer = None
        self.incorrect_number = None
        self.total_number = None
        self.labels = None
        self.buggy_patched = None

    def add_data(self,new_data, new_labels, new_buggy_patched):
        if not new_data.any():
            print('Null new dataset')
            raise
        self.dataset = np.concatenate((self.dataset, new_data), axis=0)
        self.labels = np.concatenate((self.labels, new_labels))
        self.buggy_patched = np.concatenate((self.buggy_patched, new_buggy_patched), axis=0)
        self.total_number = len(self.dataset)

    def deduplicate(self,):
        index_dupli = np.unique(self.buggy_patched[:, :], return_index=True, axis=0)[1]
        print("unique number: {}".format(len(index_dupli)))

        self.dataset = self.dataset[index_dupli]
        self.labels = self.labels[index_dupli]
        self.total_number = len(self.dataset)
        self.correct_nubmer = list(self.labels).count(1)
        self.incorrect_number = list(self.labels).count(0)


