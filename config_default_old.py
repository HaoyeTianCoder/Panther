import os
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import json

class Config:
    def __init__(self):
        self.defects4j_buggy = '/Users/haoye.tian/Documents/University/project/defects4j_buggy'
        self.path_patch = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEM'
        self.tools = ['RSRepairA']

    def __init__2(self):
        self.ods_data = '/Users/haoye.tian/Downloads/ODS/data-deduplicate'
        self.ods_feature = '/Users/haoye.tian/Downloads/ODS/features'

        self.correct_patches = [os.path.join(self.ods_data,'Closure/human-closure'),
                                os.path.join(self.ods_data,'human_patches'),
                                os.path.join(self.ods_data,'PS/correct')]
        self.correct_engineering_features = [os.path.join(self.ods_feature, 'Closure/P4J/Correct'),
                                             os.path.join(self.ods_feature, 'Human_Patch/P4J'),
                                             os.path.join(self.ods_feature, 'PS/correct/P4J')
                                             ]

        self.incorrect_patches = [os.path.join(self.ods_data,'Closure/incorrect'),
                                  os.path.join(self.ods_data, 'DRR'),
                                  os.path.join(self.ods_data, 'PS/incorrect')]

        # self.incorrect_patches = [os.path.join(self.ods_data, 'DRR'),
        #                           os.path.join(self.ods_data, 'PS/incorrect')]

        self.incorrect_engineering_features = [os.path.join(self.ods_feature, 'Closure/P4J/Incorrect'),
                                             os.path.join(self.ods_feature, 'DRR/P4J'),
                                             os.path.join(self.ods_feature, 'PS/incorrect/P4J')
                                             ]

        # self.incorrect_engineering_features = [os.path.join(self.ods_feature, 'DRR/P4J'),
        #                                        os.path.join(self.ods_feature, 'PS/incorrect/P4J')]