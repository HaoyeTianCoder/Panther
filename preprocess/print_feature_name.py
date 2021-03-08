import os
import json

path_json = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEMUnique/Developer/Correct/Lang/15/patch1/features_patch1-Lang-15-Developer-s.java->patch1-Lang-15-Developer-t.java.json'

other_vector = []
P4J_vector = []
repair_patterns = []
repair_patterns2 = []

with open(path_json, 'r') as f:
    feature_json = json.load(f)
    features_list = feature_json['files'][0]['features']
    P4J = features_list[:-2]
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
    for i in range(len(P4J)):
        dict = P4J[i]
        k = list(dict.keys())[0]
        P4J_vector.append(k)

    # repair pattern
    for k, v in RP['repairPatterns'].items():
        repair_patterns.append(k)

    # repair pattern 2
    for k, v in RP2.items():
        repair_patterns2.append(k)

print(P4J_vector + repair_patterns + repair_patterns2)