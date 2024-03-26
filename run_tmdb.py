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

    results = []  # save the results

    for line in tqdm(data):
        logger.info(f'Query: {line["query"]}')
        logger.info(f'Query: {line["solution"]}') # The most simple setting for intuitive illustration. You can add more tools based on your own setting
        tools = [name2doc[tool] for tool in line['solution']]

        # initialize the three agents
        p_model = GroAgent(model_name, tools, url=url)
        a_model = ExecAgent(model_name, tools, requests_wrapper=requests_wrapper, url=url)
        o_model = ObseAgent(model_name, tools, url=url)
        # try:
        result = _run(line, p_model, a_model, o_model)
        results.append(result)
        # except Exception as e:
        #     print(f'error...{e}')

    return results


def _run(example, p_model: GroAgent, a_model: ExecAgent, o_model: ObseAgent):
    result = []
    background = example['query']
    hidden = 'Thought: '
    for i in range(5):
        # ground the user's instruction into a tool-use instruction
        thought, action = p_model.generate(query=example['query'], hidden=hidden, api_type='tmdb')
        if thought.lower() == 'finish':
            print(thought, '\n', action)
            break
        logger.info('GroAgent: ' + thought + '->' + action)

        # execution the tool-use instruction and get the execution results
        input_request, instruction, response = a_model.generate(thought=thought, action=action, background=background)

        # observe the useful information from the length execution results
        observation = o_model.generate(instruction=instruction, json_request=input_request, response=response)
        observation = observation.strip().replace('\n', ' ')

        # update the context and incorporate the extracted information into the next action prediction
        background += ' -> ' + observation
        hidden += f'{thought}\nAPI Selection: {action}\nExecuted Result: {observation}\nThought: '
        result.append([thought,action,observation])
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
    print(headers)
    data = load_data(args.data_file)['data']
    tools = load_data(args.data_file)['tool']

    results = run(model_name=args.model_name, headers=headers, endpoints=tools, data=data, url=None)
    write_file(results, args.log_file)
