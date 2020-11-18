import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import json

class Config:
    def __init__(self):
        self.defects4j_buggy = '/Users/haoye.tian/Documents/University/project/defects4j_buggy'
        self.path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2UniqueTokenTest'
        self.dataset_name = 'PatchCollectingV2UniqueTokenTest'
        self.wcv = 'Bert'
        self.learned_feature = ''
        self.engineered_feature = ''

    def __init__2(self):
        self.ods_data = '/Users/haoye.tian/Downloads/ODS/data-deduplicate'
