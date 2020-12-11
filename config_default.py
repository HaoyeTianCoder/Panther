import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import json

class Config:
    def __init__(self):
        self.defects4j_buggy = '/Users/haoye.tian/Documents/University/project/defects4j_buggy'
        self.path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1UniqueCross'
        self.version = 'V1U'
        self.wcv = 'Bert'
        self.learned_feature = ''
        self.engineered_feature = ''

    def tmp(self):
        pass