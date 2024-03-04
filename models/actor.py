from utilize.apis import get_from_openai
from models.base import Base
import requests
import re
import logging
import json

logger = logging.getLogger()

SYSTEM_ACT = 'You are an agent that gets a sequence of APIs, which can be accessed by HTTP request. Given the API documentation and my instruction, you should complete the API call.'

ACT_PROMPT = """Please help me to complete the corresponding API calls according to the plan.  
Here is the documentation on the API:
{api_doc}

I will give you the background information, which may contain the person id, user id or movie id. You should complete the HTTP request based on according to the API documentation and current background.
The HTTP request should be a JSON string that has four basic keys: url, params, output_instructions and description.
- The value of "url" should be a string. If the API path contains "{{}}", it means that it is a variable and you should find the appropriate value from provided background and replace the '{{}}'.  For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url.
- The value of "params" should be the corresponding parameters based on the above documentation. The parameters should be extracted from the given background, e.g., user id, person id and movie id.
- The value of "output_instructions" should be instructions on what information to find from the response. Note "output_instructions" MUST be natural language and as verbose as possible! It cannot be "return the full response". Output instructions should faithfully contain the contents of the api calling plan and be as specific as possible. The output instructions can also contain conditions such as filtering, sorting, etc.
- The value of "description" should describe what the information got by this API request, e.g., the information about xxx movie, the personal information about xxx and the movie credit list of an actor. The description should be specific.

If the plan includes expressions such as "most", you should choose the first item from the response. For example, if the plan is "GET /trending/tv/day to get the most trending TV show today", you should choose the first item from the response.

Starting below, you must follow this format:
Background: background information that you can use to execute the plan, e.g., the id of a person.
Instruction: follow my instruction to complete the API calling.
API Selection: the selected API to use
HTTP request:: the HTTP request method including four basic key: url, params, output_instructions and description

Here is an example:
Background: The person id of Akira Kurosawa is 5026.
Instruction: use the person id **5026** and https://api.themoviedb.org/3/person/{{person_id}}/movie_credits to get the movies directed by Akira Kurosawa. And then extract the movies' name and id.
API Selection: https://api.themoviedb.org/3/person/{{person_id}}/movie_credits
HTTP request: {{
    "url": "https://api.themoviedb.org/3/person/5026/movie_credits",
    "params": {{
        "page": 1
    }},
    "output_instructions": "Extract the names and ids of the movies",
    "description": "The movie credit list of Akira Kurosawa."
}}

Begin!

Background: {background}
Instruction: {thought}
API Selection: {api}
HTTP request: """


class ActGPT(Base):

    def __init__(self, model='gpt-3.5-turbo', endpoints=None, requests_wrapper=None, role='ActGPT', url=None):
        super().__init__(model=model, endpoints=endpoints, role=role, url=url)
        self.requests_wrapper = requests_wrapper

    def _get_response(self, data, method=''):
        state = False
        if method == "GET":
            if 'params' in data:
                params = data.get("params", [])
                response = self.requests_wrapper.get(data.get("url"), params=params)
            else:
                response = self.requests_wrapper.get(data.get("url"))
        elif method == "POST":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.post(data["url"], params=params, data=request_body)
        elif method == "PUT":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.put(data["url"], params=params, data=request_body)
        elif method == "DELETE":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.delete(data["url"], params=params, json=request_body)
        else:
            raise NotImplementedError

        if isinstance(response, requests.models.Response):
            if response.status_code == 200:
                state = True
                response_text = response.text
            elif response.status_code == 204:
                state = True
                response_text = json.dumps({"content": "The API request has executed successfully, but the requested resource is empty."})
            else:
                response_text = response.text
        elif isinstance(response, str):
            response_text = response
        else:
            raise NotImplementedError

        return state, response_text

    def extract(self, sss):
        matches = re.findall(r'\{(.+?)\}', sss)
        return matches

    def generate(self, thought, action, background):
        api = self.match_tools(action)
        api_doc = self.get_tool_doc(api)
        prompt = ACT_PROMPT.format(api_doc=api_doc, thought=thought, background=background, api=action)
        messages = [{'role': "system", 'content': SYSTEM_ACT}, {'role': 'user', 'content': prompt}]
        response = None
        input_request = None
        for i in range(0, 3):
            input_request = get_from_openai(model_name=self.model, temp=0, messages=messages, json_mode=True, stop=['FINISH:'])['content']
            messages.append({'role': 'assistant', 'content': input_request})
            logger.info(f'Actor: {input_request}')
            data = json.loads(input_request)

            empty_value = self.extract(data['url'])
            if empty_value != []:
                messages.append({'role': 'user', 'content': "You should always change the {}" + f" in API path with the value. Specifically, please extract value from background and change the {empty_value} in your url"})
            else:
                try:
                    state, response = self._get_response(data, api['method'])
                    if state:
                        break
                    else:
                        messages.append({'role': 'user',
                                         'content': f'The API server raise a error message: {response}, please try again.'})
                except Exception as e:
                    messages.append({'role': 'user', 'content': f'There are something error {e}, please try again. You should always change the ' + "{} in API path"})
                    logger.info(f'API error: {e}')
        if response == None:
            return ['The execution result is None'] * 3
        if response == [] or response == '':
            return ['The execution result is empty'] * 3
        if input_request == None:
            return ['Please try again'] * 3
        self.add_traj(input_request)
        self.add_traj(response)
        input_request = json.loads(input_request) if type(input_request) == str else input_request
        response = json.loads(response) if type(response) == str else response
        # # parse
        instruction = input_request['output_instructions']
        return input_request, instruction, response
