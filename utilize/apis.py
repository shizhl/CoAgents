import json
import sys
from openai import OpenAI
import time
import random

api_keys_list=[
    'sk-ZmVeKnYDii42NZzVD245Af89F04b469dB9Db08C9B85aCf30'
]

def get_from_openai(model_name='gpt-3.5-turbo',base_url=None,api_key=None,
        messages=None, prompt=None,stop=None,max_len=1000, temp=1, n=1,
        json_mode=False, usage=False):
    """
    :param model_name: text-davinci-003, gpt-3.5-turbo, gpt-3.5-turbo-0613
    """
    for i in range(10):
        try:
            client = OpenAI(api_key=api_keys_list[random.randint(0, 100000) % len(api_keys_list)] if api_key==None else api_key,
                         base_url='https://api.aiguoguo199.com/v1' if base_url==None else base_url)
            kwargs={
                "model":model_name, 'max_tokens':max_len,"temperature":temp,
                "n":n, 'stop':stop, 'messages':messages,
            }
            if json_mode==True:
                kwargs['response_format']={"type": "json_object"}
            if 'instruct' in model_name:
                assert prompt!=None
                kwargs['prompt']=prompt
                response= client.completions.create(**kwargs)
            else:
                assert messages!=None
                kwargs['messages']=messages
                response= client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content if n == 1 else [res.message.content for res in response.choices]
            results={"content":content}
            if usage==False:
                results['usage']=response.usage
            return results
        except:
            error = sys.exc_info()[0]
            print("API error:", error)
            time.sleep(1)
    return 'no response from openai model...'


