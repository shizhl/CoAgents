import sys
sys.path.append('../../')
import pickle
import multiprocessing
import json
import torch
import random
import os
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from colorama import Fore

class ColorPrint:
    def __init__(self):
        self.color_mapping = {
            'Query':Fore.WHITE,
            'Ground truth':Fore.WHITE,
            "Planner": Fore.GREEN,
            "Actor": Fore.YELLOW,
            "Parser": Fore.BLUE,
            "Error": Fore.RED,
        }

    def write(self, data):
        module = data.split(':')[0]
        if module not in self.color_mapping:
            print(data, end="")
        else:
            print(self.color_mapping[module] + data + Fore.RESET, end="")

def mean(li,r=4):
    return round((sum(li))/(len(li)+0.0001),r)

def seed_torch(seed=1048):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # disable the hash random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def multi_load_jsonl(filename,num_processes=10):
    """

    :param filename: the jsonl file with big size
    :param num_processes:
    :return:
    """
    with open(filename,'r',encoding='utf-8') as f:
        data=[line.strip() for line in f]
        if len(data)<=20000:

            _,data=load_jsonl(0,data)
            return data

    length = len(data) // num_processes + 1
    pool=multiprocessing.Pool(processes=num_processes)
    collects=[]
    for ids in range(num_processes):
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(load_jsonl,(ids,collect)))

    pool.close()
    pool.join()
    results=[]
    for i,result in enumerate(collects):
        ids,res=result.get()
        assert ids==i
        results.extend(res)
    print(f"*************************** total {len(results)}  examples ****************************")
    return results

def load_jsonl(ids,data):
    data=[json.loads(line) for line in tqdm(data)]
    return ids,data

def write_file(data,filename,num_processes=20,default_name='train',indent=4):
    print(f"************************** begin to write data to {filename} *******************************")
    if filename.endswith('.json'):
        json.dump(data,open(filename,'w'),indent=indent)
    elif filename.endswith('.jsonl') :
        with open(filename, 'w') as f:
            for line in data:
                f.write(json.dumps(line) + '\n')
    elif filename.endswith('.txt'):
        with open(filename, 'w') as f:
            for line in data:
                f.write(str(line) + '\n')
    elif filename.endswith('.pkl'):
        pickle.dump(data,open(filename,'w'))
    elif '.' not in filename:
        multi_write_jsonl(data,filename,num_processes=num_processes,default_name=default_name)
    else:
        raise "no suitable function to write data"
    print(f"************************** totally {len(data)} writing data to {filename} *******************************")

def write_jsonl(data,filename,ids=None):
    with open(filename,'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line)+'\n')
    return ids,len(data)

def multi_write_jsonl(data,folder,num_processes=10,default_name='train'):
    """

    :param filename:
    :param num_processes:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    length = len(data) // num_processes + 1
    pool=multiprocessing.Pool(processes=num_processes)
    collects=[]
    for ids in range(num_processes):
        filename=os.path.join(folder,f"{default_name}{ids}.jsonl")
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(write_jsonl,(collect,filename,ids)))

    pool.close()
    pool.join()
    cnt=0
    for i,result in enumerate(collects):
        ids,num=result.get()
        assert ids==i
        cnt+=num
    print(f"** total {cnt}  examples have been writen to {folder} **")
    return cnt

def load_data(filename,num_processes=10):
    print(f"************************** begin to load data of {filename} *******************************")
    if filename.endswith('.jsonl'):
        return multi_load_jsonl(filename,num_processes)
    elif filename.endswith('.json'):
        return json.load(open(filename,'r'))
    elif filename.endswith('.pkl'):
        return pickle.load(filename)
    elif filename.endswith('.txt'):
        with open(filename,'r') as f:
            data=[line.strip() for line in f]
            return data
    else:
        raise "no suitable function to load data"


def multi_process_cuda(data_path,ranks,func,**kwargs):
    """

    :param data_path: data path
    :param ranks: gpu device id
    :param func: the function for batch
    :param kwargs: the 'dict', indicating the parameter to pass into the 'func'
    :return:
    """
    torch.multiprocessing.set_start_method('spawn',force=True)
    cuda_pool=mp.Pool(processes=len(ranks))
    data=load_data(data_path)
    length = len(data) // len(ranks) + 1
    collects=[]
    for ids,rank in enumerate(ranks):
        collect = data[ids * length:(ids + 1) * length]
        collects.append(cuda_pool.apply_async(func, (rank,collect,kwargs)))
    cuda_pool.close()
    cuda_pool.join()
    results=[]
    for rank,result in zip(ranks,collects):
        r,res=result.get()
        assert r==rank
        results.extend(res)
    return results

def multi_process_cuda_data(data,ranks,func,**kwargs):
    """

    :param data_path: the data path
    :param ranks: gpu device ids
    :param func:
    :param kwargs:
    :return:
    """
    torch.multiprocessing.set_start_method('spawn',force=True)
    cuda_pool=mp.Pool(processes=len(ranks))
    length = len(data) // len(ranks) + 1
    collects=[]
    for ids,rank in enumerate(ranks):
        collect = data[ids * length:(ids + 1) * length]
        collects.append(cuda_pool.apply_async(func, (rank,collect,kwargs)))
    cuda_pool.close()
    cuda_pool.join()
    results=[]
    for rank,result in zip(ranks,collects):
        r,res=result.get()
        assert r==rank
        results.extend(res)
    return results
