import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import json

class Config:
    def __init__(self):
        self.path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEMYeUnique'
        self.version = 'Cross_bigdata'
        self.wcv = 'Bert'
        # self.defects4j_buggy = '/Users/haoye.tian/Documents/University/project/defects4j_buggy'
        # self.path_testdata = '/Users/haoye.tian/Documents/University/data/PatchSimTOSEM'

        # self.learned_feature = ''
        # self.engineered_feature = ''