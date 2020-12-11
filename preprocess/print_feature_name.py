import os
import json

path_json = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1UniqueCross/HDRepair/Correct/Math/70/features_patch1-Math-70-HDRepair-s.java->patch1-Math-70-HDRepair-t.java.json'

other_vector = []
P4J_vector = []
repair_patterns = []
with open(path_json, 'r') as f:
    feature_json = json.load(f)
    features_list = feature_json['files'][0]['features']
    other = features_list[0]
    P4J = features_list[-2]
    RP = features_list[-1]

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
    for k, v in P4J.items():
        k = k[1:-1]
        k = '-'.join(k.split(', '))
        P4J_vector.append(k)

    # repair pattern
    for k, v in RP['repairPatterns'].items():
        repair_patterns.append(k)

print(P4J_vector + repair_patterns)