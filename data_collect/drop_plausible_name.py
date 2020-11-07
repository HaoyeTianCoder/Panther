import os

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'
# tools = ['VFix','AVATAR', 'FixMiner', 'TBar', 'kPAR', 'jKali', 'jMutRepair', 'Cardumen',
#          'GenProg-A', 'Kali-A', 'RSRepair-A']

# for tool in tools:
#     for root, dirs, files in os.walk(os.path.join(path, tool)):
#         for file in files:
#             label = root.split('/')[-2]
#             if label == 'Correct':
#                 if file.endswith('.buggy') or file.endswith('.fixed') or file.endswith('.patch'):
#                     old = os.path.join(root, file)
#
#                     # new = file.split('.')[0] + '-plausible.' + file.split('.')[1]
#                     # new = os.path.join(root, new)
#
#                     new = old.replace('-plausible','')
#                     print(new)
#                     os.rename(old, new)

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.buggy') or file.endswith('.fixed') or file.endswith('.patch'):
            old = os.path.join(root, file)

            new = old.replace('-plausible','')
            print(old)
            print(new)
            os.rename(old,new)