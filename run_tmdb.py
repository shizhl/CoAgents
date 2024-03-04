from langchain.requests import Requests
from models.actor import ActGPT
import logging
from utilize.utilze import ColorPrint, load_data, write_file, mean
from models.parse import ParseGPT
from tqdm import tqdm
from models.plan import *


logger = logging.getLogger()

class Interact:

    @staticmethod
    def run(data, model_name='gpt-3.5-turbo', endpoints=None,url=None):
        logging.basicConfig(
            format="%(message)s",
            handlers=[logging.StreamHandler(ColorPrint())],
        )
        logger.setLevel(logging.INFO)
        access_token='eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwZGJhYjU5MGM3ZWFjYTA3ZWJlNjI1OTc0YTM3YWQ5MiIsInN1YiI6IjY1MmNmODM3NjYxMWI0MDBmZmM3MDM5OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.McsK4Wm5XnRSDLn62Jhy787YUAwZcQz0X5qzkGuLe_s'
        headers = {'Authorization': f'Bearer {access_token}' }
        requests_wrapper = Requests(headers=headers)

        name2doc = {line['url']: line for line in endpoints}
        results=[]
        for line in tqdm(data):
            logger.info(f'Query: {line["query"]}')
            logger.info(f'Query: {line["solution"]}')
            tools = [tool for tool in line['solution']]
            tools = [e.replace(' ',' https://api.themoviedb.org/3') for e in tools if e not in line['noise']]
            tools = [name2doc[tool] for tool in tools]
            p_model = PlanGPT(model_name, tools,url=url)
            a_model = ActGPT(model_name, tools, requests_wrapper=requests_wrapper,url=url)
            o_model = ParseGPT(model_name,tools,url=url)
            for i in range(0,3):
                try:
                    line['result']=Interact._run(line, p_model, a_model,o_model)
                    line['traj']= {"plan":p_model.traj,'act':a_model.traj,'parse':o_model.traj}
                    line['token']= {"plan":sum(p_model.token),'act':sum(a_model.token),'parse':sum(o_model.token)}
                    print('the token is '+str(line['token']))
                    break
                except:
                    print(f'try {i}...')
                    line['token']= None
                    line['result'] = None
            results.append(line)
        return results

    @staticmethod
    def _run(line,p_model:PlanGPT,a_model:ActGPT,o_model:ParseGPT):
        result = []
        background = line['query']
        hidden = 'Thought: '
        for i in range(5):
            thought, action, obs = p_model.generate(query=line['query'], hidden=hidden, api_type='tmdb')
            if obs=='FINISH':
                print(thought)
                print(action)
                result.append({"thought": thought, 'action': action, 'observation':obs})
                break
            logger.info('Planner: ' + thought)

            input_request,instruction,response = a_model.generate(thought=thought, action=action, background=background)
            observation= o_model.generate(instruction=instruction,json_request=input_request, response=response)
            observation = observation.strip().replace('\n', ' ')

            background += ' -> ' + observation
            result.append({"thought": thought, "action": action, "observation": observation})
            hidden += f'{thought}\nAPI Selection: {action}\nExecuted Result: {observation}\nThought: '
        return result


def evaluate(log_file):
    data=load_data(log_file)
    recall=[]
    precise=[]
    success_rate=[]
    base_url='https://api.themoviedb.org/3'
    for line in data:
        if line['result']==None:
            recall.append(0)
            precise.append(0)
            break
        pred=[e['action'].replace(base_url,'') for e in line['result'] if e['observation']!='FINISH']
        overlap=len([e for e in line['solution'] if e[4:] in pred])
        success_rate.append(overlap==len(line['solution']))
        precise.append(overlap/len(pred))
        recall.append(overlap/len(line['solution']))

    return mean(success_rate),mean(precise),mean(recall)

if __name__ == '__main__':
    log_file='./logs/tmdb.token.json'
    data=load_data('./dataset/tmdb.data.tool.json')['data']
    tools=load_data('./dataset/tmdb.data.tool.json')['tool']

    Interact.run(model_name='gpt-3.5-turbo', # gpt-3.5-turbo
                 endpoints=tools, data=data,)
    result=evaluate(log_file)
    print(result)


