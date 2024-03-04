import json
from utilize.utilze import *
from typing import List
from scipy.stats import ttest_ind


def longestCommonSubsequence(text1: List[str], text2: List[str]) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
def normalize(sss):
    for e in ['https://api.themoviedb.org/3', 'GET']:
        sss = sss.replace(e, '').strip()
    for i in range(10):
        sss = sss.replace(f'[{i}]', '').strip()
    return sss.strip()


def overlap(s1, s2):
    s1 = s1.split('/')
    s1 = [e.strip() for e in s1]
    s2 = s2.split('/')
    s2 = [e.strip() for e in s2]
    return len([e for e in s1 if e in s2])

def match_tools(api_name):
    tool = json.load(open('/Users/shizhl/Paper2024/LearnInteract/NovelTools/tmdb.data.tool.json'))['tool']
    urls = [line['url'] for line in tool]
    res = [(overlap(url, api_name), url) for url in urls]
    res = sorted(res, key=lambda x: x[0], reverse=True)
    return res[0][-1]

def api_eval(preds, answers,avg=True):
    recall = []
    precise = []
    success_rate = []
    delta = []
    f1 = []
    correct_path = []
    answers = [[e.lower() for e in line] for line in answers]
    preds = [[e.lower() for e in line] for line in preds]
    for pred, answer in zip(preds, answers):
        # print('-'*10)
        # print(pred)
        # print(answer)
        overlap = len([e for e in answer if e in pred])
        p = overlap / (len(pred) + 0.001)
        r = overlap / (len(answer) + 0.001)
        success_rate.append(overlap == len(answer) * (pred[-1].lower() == 'finish'))  # * (pred[-1] == 'finish')
        precise.append(p)
        recall.append(r)
        correct_path.append(longestCommonSubsequence(pred, answer) / len(answer))
        delta.append(max((len(pred) - len(answer)), 0) * success_rate[-1])  # /len(answer)
        f1.append(2 * p * r / (p + r + 0.0001))
    # delta = [e for e in delta if e != 0]
    if avg:
        return mean(success_rate), mean(f1), mean(precise), mean(recall), mean(delta), mean(correct_path)
    else:
        return  success_rate,f1, precise,recall, delta, correct_path
def toolbench(log_file):
    data = json.load(open(log_file))
    # preds=[[api['content'] for api in line['result'] if api['role']=='plan'] for line in data]
    preds = []
    for line in data:
        pred=[api[0]['content'] for api in line['result']]
        if line['win'] and pred[-1].lower()!='finish':
            pred.append('finish')
        elif line['win']==False:
            pred.append('unfinish')
        preds.append(pred)
    answers = [line['solution']+['finish'] for line in data]
    sr, f1, p, r, delta, cp = api_eval(preds=preds, answers=answers)
    # token

    print(sr, f1, p, r, delta, cp)

def t_test(file1,file2):
    def func(l1,l2):
        t_stat, p_value = ttest_ind(l1, l2)
        print(f"t统计量: {t_stat}")
        print(f"P值: {p_value}")

        # 根据P值做出决策
        if p_value < 0.05:
            print("两组样本均值存在显著差异（拒绝零假设）")
        else:
            print("两组样本均值没有显著差异（不拒绝零假设）")

    data1 = json.load(open(file1))
    data2 = json.load(open(file2))
    results=[]
    for line1 in data1:
        for line2 in data2:
            if line1['query']==line2['query'] and  line1['result']!=None and  line2['result']!=None:
                res1=[normalize(api['action']) for api in line1['result']] + ['Finish']
                res2=[normalize(api['action']) for api in line2['result']] + ['Finish']
                results.append(([e.replace('GET ', '') for e in line1['solution']]+ ['Finish'] ,res1,res2))
    f1_1,sr_1,_,_,_,_=api_eval([e[1] for e in results],[e[0] for e in results],avg=False)
    f1_2,sr_2,_,_,_,_=api_eval([e[2] for e in results],[e[0] for e in results],avg=False)
    f1_1=[int(e) for e in f1_1]
    f1_2=[int(e) for e in f1_2]
    func(f1_1,f1_2)
    func(sr_1,sr_2)


def ours(log_file):
    data = json.load(open(log_file))
    print(len(data))
    preds = [[normalize(api['action']) for api in line['result']] + ['Finish'] for line in data if line['result']!=None]
    answers = [[e.replace('GET ', '') for e in line['solution']] + ['Finish'] for line in data if line['result']!=None]
    sr, f1, p, r, delta, cp = api_eval(preds=preds, answers=answers)
    print(sr, f1, p, r, delta, cp)

def offline(log_file):
    data = json.load(open(log_file))
    preds=[]
    for line in data:
        if line['result'] == None:
            pred=['Unfinish']
        else:
            pred = [normalize(api['action']) for api in line['result']]
            if line['win']:
                pred.append('finish')
            else:
                pred.append('unfinish')
        preds.append(pred)
    answers = [[e.replace('GET ', '') for e in line['solution']]+['finish'] for line in data]

    sr, f1, p, r, delta, cp = api_eval(preds=preds, answers=answers)
    print(sr, f1, p, r, delta, cp)



def restbench(log_file):
    data = json.load(open(log_file))
    raw = json.load(open('/Users/shizhl/Paper2024/LearnInteract/NovelTools/tmdb.data.tool.json'))['data']
    query2solution = {line['query'].strip(): line['solution'] for line in raw}
    preds = []
    answers = []
    for line in tqdm(data):
        # print(line['query'])
        if 'traj' not in line:
            print(line)
            continue
        tmp = []
        for api in line['traj']:
            if 'action' in api and 'FINISH' not in api['thought']:
                url = json.loads(api['action'])['url'] if type(api['action']) == str else api['action']['url']
                tmp.append(match_tools(normalize(url)))
                # print(url,tmp[-1])
        preds.append(tmp+['finish'])
        answers.append(query2solution[line['query']]+['finish'])

    sr, f1, p, r, delta, cp = api_eval(preds=preds, answers=answers)
    print(sr, f1, p, r, delta, cp)


if __name__ == '__main__':
    t_test('/Users/shizhl/Paper2024/LearnInteract/logs/tmdb.0.100.json',
           '/Users/shizhl/Paper2024/LearnInteract/logs/tmdb.token.gpt4.json')
    data = json.load(open('/Users/shizhl/Paper2024/LearnInteract/NovelTools/tmdb.data.tool.json'))['data']
    files = ['/Users/shizhl/Paper2024/Toolbench/atmdb/react.json',
             '/Users/shizhl/Paper2024/Toolbench/atmdb/DFS_woFilter_w2.json',
             '/Users/shizhl/Paper2024/LearnInteract/logs/tmdb.0.100.json',
             '/Users/shizhl/Paper2024/LearnInteract/offline/tmdb.output-mixtral.json',
             '/Users/shizhl/Paper2024/RestGPT/restbench-tmdb.v1.json',
             '/Users/shizhl/Paper2024/LearnInteract/offline/tmdb.output.ours.json',
             '/Users/shizhl/Paper2024/Toolbench/atmdb/react@3_0.json',]
    # print('react')
    # toolbench(files[0])
    # print('react@3')
    # toolbench(files[6])
    # # toolbench(files[7])
    # # toolbench(files[8])
    # print('toolllama')
    # toolbench(files[1])
    # print('ours')
    ours(files[2])
    # ours('/Users/shizhl/Paper2024/LearnInteract/logs/tmdb.token.gpt4.json')
    # ours('/Users/shizhl/Paper2024/LearnInteract/logs/gpt.wo.interact.tmdb.json')
    # ours('/Users/shizhl/Paper2024/LearnInteract/logs/gpt.wo.parse.tmdb.json')
    # print('Chameleon ')
    # offline(files[3])
    # print('RestGPT')
    # restbench(files[4])
    # print('Chameleon + ours')
    # offline(files[5])
    #
    # ours('/Users/shizhl/Paper2024/LearnInteract/logs/mistral87.tmdb.json')


