import os
import matplotlib.pyplot as plt

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2'

tools = os.listdir(path)
tools.remove('.DS_Store')
share = []

total = 0
for tool in tools:
    if tool.startswith('.'):
        continue
    cnt = 0
    for root, dirs, files in os.walk(os.path.join(path, tool)):
        for file in files:
            if file.endswith('.patch'):
                cnt += 1
                total += 1
    share.append(cnt)

print(total)

plt.figure(figsize=(20, 8), dpi=100)

# plt.pie(share, labels=tools, autopct="%1.1f%%",)
# plt.pie(share, labels=tools,)
plt.pie([48481, 13716, 2096], labels=['Arja','GenProg','Others'],)

# plt.legend()

plt.title("tools share")

plt.show()

