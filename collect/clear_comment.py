import os

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/ACS'

cnt = 0
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            file_path = os.path.join(root, file)
            new_line = ''
            with open(file_path,'r') as f:
                for line in f:
                    if line.startswith('+'):
                        if line[1:].strip() == "//ACS's patch begin" or line[1:].strip() == "//ACS's patch end":
                            continue
                    if line.startswith('\ No newline at end of file') or line.startswith('Common subdirectories'):
                        continue
                    if line.startswith('@@'):
                        tmp1 = line.split(' ')
                        tmp2 = tmp1[2].split(',')
                        num = int(tmp2[1])
                        num -= 2
                        fixed_num = tmp2[0]+ ',' + str(num)
                        tmp1[2] = fixed_num
                        final = ' '.join(tmp1)
                        new_line += final
                    else:
                        new_line += line
            new_file = open(file_path,'w')
            new_file.write(new_line)
            new_file.close()
            cnt += 1
print(cnt)