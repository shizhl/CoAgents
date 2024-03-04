from utilize.apis import *

class Base:

    def __init__(self,model,endpoints,role='base',url=None):
        self.role=role
        self.model=model
        self.endpoints=endpoints
        self.url=url
        if endpoints!=None:
            self.name2doc = {line['url']: line for line in endpoints}
            base_url=endpoints[0]['base_url']
            for line in endpoints:
                self.name2doc[f"{base_url}{line['url']}"]=line
        else:
            self.name2doc=None
        self.traj=[]
        self.token=[]

    def add_traj(self, output):
        self.traj.append({'role':self.role,'content':output})

    def overlap(self,s1,s2):
        s1=s1.split('/')
        s1=[e.strip() for e in s1]
        s2=s2.split('/')
        s2 = [e.strip() for e in s2]
        return len([e for e in s1 if e in s2 ])

    def match_tools(self,input_api):
        api_name=self.normalize(input_api)
        if api_name.startswith("'") or api_name.startswith('"'):
            api_name=api_name[1:]
        if api_name.endswith("'") or api_name.endswith('"'):
            api_name = api_name[:-1]
        if "?" in api_name:
            api_name=api_name[:api_name.index('?')].replace('?','').strip()
        if api_name in self.name2doc:
            return self.name2doc[api_name]

        api_list=[line['method']+ ' ' + line['url'] for line in self.endpoints]
        api_list='\n'.join(api_list)
        prompt=f"I mistakenly remembered the name of an API, and please help me to choose the most likely API from the provided list.\nInput API: {input_api}\nAPI list: {api_list}\n\nYou should only select a API from list, and do not say extra words."
        api_name=get_from_openai(model_name=self.model,messages=[{'role':'user','content':prompt}])['content'].strip()

        if api_name in self.name2doc:
            return self.name2doc[api_name]
        #
        res=[(self.overlap(name,api_name),v) for name,v in self.name2doc.items()]
        res=sorted(res,key=lambda x:x[0],reverse=True)
        return res[0][-1]

    def normalize(self,sss):
        ee1=['Thought:','API Selection:','Execution Result:']
        ee2=['\n']
        for e in ee1:
            sss=sss.replace(e,'')
        for e in ee2:
            sss=sss.replace(e,' ')
        for i in range(10):
            sss=sss.replace(f'[{i}]','')
        return sss.strip()

    def get_tool_doc(self,line):
        url = f'{line["base_url"]}/{line["url"]}' if 'http' not in line['url'] else line['url']
        description = '## Description\n' + line['description'].replace('\n',' ')
        if  len(line['para'])==0:
            para='## Parameter:\nNo parameter'
        else:
            para = ['## Parameter']
            for p in line['para']:
                if p['in'] !='query':
                    continue
                tmp="- "+p['name']+": "+p['description'].replace('\n','')
                if 'schema' in p and 'type' in p['schema']:
                    tmp+=" (type: "+p['schema']['type']+")"
                para.append(tmp)
            para='\n'.join(para)

        prompt = [url, description, para]
        if 'usage' in line and line['usage'] != '':
            usage = '## Usage:\n```python\n' + line['usage'] + '\n```'
            prompt.append(usage)
        if 'result' in line and line['result'] != '':
            result = '## Execution result:\n```json\n' + json.dumps(line['result'], indent=4) + '\n```'
            prompt.append(result)

        prompt = '\n'.join(prompt)
        return prompt