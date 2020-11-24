import os
import matplotlib.pyplot as plt

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2'

tools = os.listdir(path)
tools.remove('.DS_Store')
share = []

def all():
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

    plt.figure(figsize=(20, 8), dpi=100)

    # plt.pie(share, labels=tools, autopct="%1.1f%%",)
    plt.pie(share, labels=tools,)

    # plt.legend()
    plt.title("Tools share")

    plt.show()

def correct():
    total = 0
    final_tools = []
    for tool in tools:
        if tool.startswith('.'):
            continue
        path_correct = os.path.join(path, tool) + '/Correct'
        if not os.path.exists(path_correct):
            continue
        final_tools.append(tool)
        cnt = 0
        for root, dirs, files in os.walk(path_correct):
            for file in files:
                if file.endswith('.patch'):
                    cnt += 1
                    total += 1
        share.append(cnt)

    plt.figure(figsize=(20, 8), dpi=100)

    # plt.pie(share, labels=tools, autopct="%1.1f%%",)
    plt.pie(share, labels=final_tools,)

    # plt.legend()
    plt.title("Percentage of tools in correct patches")

    plt.show()

def incorrect():
    total = 0
    final_tools = []
    for tool in tools:
        if tool.startswith('.'):
            continue
        path_incorrect = os.path.join(path, tool) + '/Incorrect'
        if not os.path.exists(path_incorrect):
            continue
        final_tools.append(tool)
        cnt = 0
        for root, dirs, files in os.walk(path_incorrect):
            for file in files:
                if file.endswith('.patch'):
                    cnt += 1
                    total += 1
        share.append(cnt)

    plt.figure(figsize=(20, 8), dpi=100)

    # plt.pie(share, labels=tools, autopct="%1.1f%%",)
    plt.pie(share, labels=final_tools,)

    # plt.legend()
    plt.title("Percentage of tools in incorrect patches")

    plt.show()

def repairThemAll():
    plt.figure(figsize=(20, 8), dpi=100)
    plt.pie([48481, 13716, 2096], labels=['Arja','GenProg','Others'],)
    plt.title("Tools share")
    plt.show()

if __name__ == '__main__':
    # correct()
    incorrect()
