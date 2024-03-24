from langchain.requests import Requests
import logging
from utilize.utilze import ColorPrint, load_data, write_file, mean
from models.execution import ExecAgent
from models.observation import ObseAgent
from models.grounding import GroAgent
from tqdm import tqdm
import argparse

logger = logging.getLogger()


def run(data, model_name, headers, endpoints, url):
    # initialize the environment
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

    # the headers used to request the API servers
    requests_wrapper = Requests(headers=headers)

    name2doc = {line['url']: line for line in endpoints}

    results = [] # save the results

    for line in tqdm(data):
        logger.info(f'Query: {line["query"]}')
        logger.info(f'Query: {line["solution"]}')
        tools = [name2doc[tool] for tool in line['solution']]

        # initialize the three agents
        p_model = GroAgent(model_name, tools, url=url)
        a_model = ExecAgent(model_name, tools, requests_wrapper=requests_wrapper, url=url)
        o_model = ObseAgent(model_name, tools, url=url)

        # solve the tasks via the cooperation and interaction of the three agents
        for i in range(0, 3):
            try:
                line['result'] = _run(line, p_model, a_model, o_model)
                line['traj'] = {"plan": p_model.traj, 'act': a_model.traj, 'parse': o_model.traj}
                line['token'] = {"plan": sum(p_model.token), 'act': sum(a_model.token), 'parse': sum(o_model.token)}
                print('the token is ' + str(line['token']))
                break
            except:
                print(f'try {i}...')
                line['token'] = None
                line['result'] = None

        results.append(line)
    return results


def _run(example, p_model: GroAgent, a_model: ExecAgent, o_model: ObseAgent):
    result = [] # save the trajectory
    background = example['query']
    hidden = 'Thought: '
    for i in range(5):
        # ground the user's instruction into a tool-use instruction
        thought, action, obs = p_model.generate(query=example['query'], hidden=hidden, api_type='tmdb')
        if obs == 'FINISH':
            print(thought,'\n', action)
            result.append({"thought": thought, 'action': action, 'observation': obs})
            break
        logger.info('GroAgent: ' + thought)

        # execution the tool-use instruction and get the execution results
        input_request, instruction, response = a_model.generate(thought=thought, action=action, background=background)

        # observe the useful information from the length execution results
        observation = o_model.generate(instruction=instruction, json_request=input_request, response=response)
        observation = observation.strip().replace('\n', ' ')

        # update the context and incorporate the extracted information into the next action prediction
        background += ' -> ' + observation
        result.append({"thought": thought, "action": action, "observation": observation})
        hidden += f'{thought}\nAPI Selection: {action}\nExecuted Result: {observation}\nThought: '
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', required=False)
    parser.add_argument('--log_file', type=str, help='your log file to save the output trajectory', required=True)
    parser.add_argument('--data_file', type=str, help='your data file containing the test examples and tools', required=True)
    parser.add_argument('--access_token_file', type=str, help='the file containing the access token required by TMDB', required=True)

    args = parser.parse_args()

    with open(args.access_token_file) as f:
        access_token = f.read().strip()
    headers = {'Authorization': f'Bearer {access_token}'}

    data = load_data(args.data_file)['data']
    tools = load_data(args.data_file)['tool']

    run(model_name=args.model_name, headers=headers, endpoints=tools, data=data, url=None)
