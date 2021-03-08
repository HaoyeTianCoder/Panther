import os
import json

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/PraPR'

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('patch'):
            project = root.split('/')[-2]
            id = root.split('/')[-1]
            result = ''
            with open(os.path.join(root, file), 'r+') as f:
                for line in f:
                    if line.startswith('@@'):
                        location = int(line.split(' ')[1].split(',')[0][1:])
                        bug_location_line = location
                        if project == 'Chart':
                            if id == '20':
                                pass
                            else:
                                bug_location_line -= 1

                        elif project == 'Closure':
                            if id == '10':
                                pass
                            elif id == '14':
                                bug_location_line += 2
                            elif id == '70':
                                bug_location_line += 1
                            elif id == '86':
                                bug_location_line -= 4
                            else:
                                bug_location_line -= 1

                            if id == '63':
                                bug_location_line += 1

                        elif project == 'Lang':
                            if id == '10':
                                bug_location_line += 5
                            elif id == '26':
                                bug_location_line += 2
                            else:
                                bug_location_line -= 1

                        elif project == 'Math':
                            if id == '70':
                                bug_location_line -= 2
                            if id == '85':
                                pass
                            else:
                                bug_location_line -= 1

                        elif project == 'Mockito':
                            bug_location_line -= 1

                        elif project == 'Time':
                            bug_location_line -= 1

                        new_line = line.replace(str(location), str(bug_location_line))
                        result += new_line
                    else:
                        result += line

            with open(os.path.join(root, file), 'r+') as f:
                f.write(result)
