from io import StringIO
import tiktoken
from utilize.apis import get_from_openai
from models.base import Base
from pydantic import BaseModel, Field
import json
from typing import Optional, Dict
import sys
import logging

logger = logging.getLogger()

encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')


def is_stop(x, length=1000):
    input_length = len(encoder.encode(x))
    print(input_length)
    if input_length < length:
        return True
    return False


LLM_PARSING_TEMPLATE = """Here is an API JSON response with its corresponding Http request:

The Http request, including url, description and parameters.
{request}

The response is about: {response}

====
Your task is to extract some information according to these instructions: {instruction}
When working with API objects, you should usually use ids over names.
If the response indicates an error, you should instead output a summary of the error.

Output:
"""

CODE_PARSING_SCHEMA_TEMPLATE1 = """Here is an API response obtained from a HTTP request, which is stored in a python dict variable called 'data', your task is to generate Python code to extract information I need from the API response.
Note: I will give you 'data', do not make up one, just reference it in your code.
Please print the final result as brief as possible. If the result is a list, just print it in one sentence. Do not print each item in a new line.
The example result format are:
"The release date of the album is 2002-11-03"
"The id of the person is 12345"
"The movies directed by Wong Kar-Wai are In the Mood for Love (843), My Blueberry Nights (1989), Chungking Express (11104)"
Note you should generate only Python code.
DO NOT use fields that are not in the response schema.

Here is the Http request, including url, description and parameters.
{request}

Response JSON format of HTTP request result:
{response_schema}

Instruction: {instruction}

The code you generate should satisfy the following requirements:
1. The code you generate should contain the filter in the query. For example, if the query is "what is the name and id of the director of this movie" and the response is the cast and crew for the movie, instead of directly selecting the first result in the crew list (director_name = data['crew'][0]['name']), the code you generate should have a filter for crews where the job is a "Director" (item['job'] == 'Director').
2. If the response is something about X, e.g., the movies credits of Lee Chang-dong, then the filter condition cannot include searching for X (e.g., Lee Chang-dong). For example, the API response is the movie credits of Akira Kurosawa and the instruction is what are the ids of the movies directed by him. Then the your code should not contain "movie['title'] == 'Akira Kurosawa'" or "movie['name'] == 'Akira Kurosawa'"
3. Do not use f-string in the print function. Use "format" instead. For example, use "print('The release date of the album is {{}}'.format(date))" instead of "print(f'The release date of the album is {{date}}')
4. Please print the final result as brief as possible. If the result is a list, just print it in one sentence. Do not print each item in a new line.
5. If the instruction includes expressions such as "most", you should choose the first item from the response. For example, if the plan is "GET /trending/tv/day to get the most trending TV show today", you should choose the first item from the response.

Begin and complete the [Python code]!
Python Code:
```python
[Python code]
```
"""

CODE_PARSING_SCHEMA_TEMPLATE = """Here is an API response obtained from a HTTP request, which is stored in a python dict variable called 'response', your task is to generate Python code to extract information I need from the API response.
Note: I will give you 'response', which has been load as a Python dict. Do not make up one, just reference it in your code. DO NOT use fields that are not in the variable `response`.

Here is the details of `response`.
## Variable Description
{description}
## Variable Structure
{response_schema}
This structure is a simplified representation of a Python dictionary, showing the type of each key’s value: 
If the value is of a basic type (such as integer, string, boolean, etc.), it is directly represented as the corresponding type (like int, str, bool, etc.).
If the value is of list type, it is represented as a list followed by square brackets [], and inside the brackets is the type of elements in the list.
If the value is of dictionary type, it is represented as a dict, and inside the curly brackets {{}} the key-value pairs of the dictionary are further broken down.

You should read the structure of the variable `response` carefully and write code based on my instruction:
Instruction: {instruction}

The code you generate should satisfy the following requirements:
1. The code you generate should contain the filter in the query. For example, if the query is "what is the name and id of the director of this movie" and the response is the cast and crew for the movie, instead of directly selecting the first result in the crew list (director_name = data['crew'][0]['name']), the code you generate should have a filter for crews where the job is a "Director" (item['job'] == 'Director').
2. Do not use f-string in the print function. Use "format" instead. For example, use "print('The release date of the album is {{}}'.format(date))" instead of "print(f'The release date of the album is {{date}}')
3. If the instruction includes expressions such as "most", you should choose the first item from the response. For example, if the plan is "GET /trending/tv/day to get the most trending TV show today", you should choose the first item from the response.
4. Please print the final result as brief as possible. If the result is a list, just print it in one sentence. Do not print each item in a new line.
The example result format are:
"The release date of the album is 2002-11-03"
"The id of the person is 12345"
"The movies directed by Wong Kar-Wai are In the Mood for Love (843), My Blueberry Nights (1989), Chungking Express (11104)"
Note you should generate only Python code.

Begin and complete the [Python code]!
Python Code:
```python
[Python code]
```"""

CODE_PARSING_SCHEMA_TEMPLATE2 = """Here is an API response obtained from a HTTP request, which is stored in a python dict variable called 'response', your task is to generate Python code to extract information I need from the API response.
Note: I will give you 'response', which has been load as a Python dict. Do not make up one, just reference it in your code. DO NOT use fields that are not in the variable `response`.

Here is the details of `response`.
## Variable Description
{description}
## Variable Structure
{response_schema}
This structure is a simplified representation of a Python dictionary, showing the type of each key’s value: 
If the value is of a basic type (such as integer, string, boolean, etc.), it is directly represented as the corresponding type (like int, str, bool, etc.).
If the value is of list type, it is represented as a list followed by square brackets [], and inside the brackets is the type of elements in the list.
If the value is of dictionary type, it is represented as a dict, and inside the curly brackets {{}} the key-value pairs of the dictionary are further broken down.

You should read the structure of the variable `response` carefully and write code based on my instruction:
Instruction: {instruction}

The code you generate should satisfy the following requirements:
1. The code you generate should contain the filter in the query. For example, if the query is "what is the name and id of the director of this movie" and the response is the cast and crew for the movie, instead of directly selecting the first result in the crew list (director_name = data['crew'][0]['name']), the code you generate should have a filter for crews where the job is a "Director" (item['job'] == 'Director').
2. Do not use f-string in the print function. Use "format" instead. For example, use "print('The release date of the album is {{}}'.format(date))" instead of "print(f'The release date of the album is {{date}}')
3. If the instruction includes expressions such as "most", you should choose the first item from the response. For example, if the plan is "GET /trending/tv/day to get the most trending TV show today", you should choose the first item from the response.
4. Please print the final result as brief as possible. If the result is a list, just print it in one sentence. Do not print each item in a new line.
The example result format are:
"The release date of the album is 2002-11-03"
"The id of the person is 12345"
"The movies directed by Wong Kar-Wai are In the Mood for Love (843), My Blueberry Nights (1989), Chungking Express (11104)"
Note you should generate only Python code.

Begin and complete the [Python code]!
Python Code:
```python
[Python code]
```"""

SYSTEM_PARSE = 'You are CodeGPT, a intelligent assistant which can write Python code to help me extract key information from a complex result of Http requests.'


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    def run(self, command: str):
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        status = True
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            # print(str(e))
            output = '\n'.join([
                '-----------------------------',
                f'except Exception as e - Exception Type: {type(e)}',
                f'except Exception as e - Exception Value: {e}'
            ])
            status = False
        return status, output


def simplify_json(json_data):
    simplified_data = {}

    for key, value in json_data.items():
        if isinstance(value, dict):
            # 递归处理嵌套的字典
            simplified_data[key] = simplify_json(value)
        elif isinstance(value, list) and value:
            # 对于非空数组，只保留第一个元素，并在类型表示中添加标记
            simplified_data[key] = [simplify_json(value[0])]
        else:
            # 用值的类型表示原始值
            simplified_data[key] = {"type": str(type(value).__name__)}

    return simplified_data


def get_yaml(value, name, indent=0):
    result = ['\t' * indent + f"- {name}: {type(value).__name__}"]
    if list == type(value):
        element = f'{name}[0]'
        indent += 1
        if value != []:
            result.extend(get_yaml(value[0], element, indent))

    elif dict == type(value):
        for k, v in value.items():
            result.extend(get_yaml(v, k, indent + 1))
    return result


class ParseGPT(Base):
    def __init__(self, model='gpt-3.5-turbo', endpoints=None, role='ParseGPT', url=None):
        super().__init__(model=model, endpoints=endpoints, role=role, url=url)

    def generate(self, instruction, json_request, response):
        response = json.dumps(response)
        if len(response.split()) < 1000:
            extract_prompt = LLM_PARSING_TEMPLATE.format(request=json_request, instruction=instruction,
                                                         response=json.dumps(response)[:1500])
            res = get_from_openai(model_name=self.model, temp=0.5, messages=[{"role": "user", 'content': extract_prompt}])['content']
            res = res.replace('\n', '').replace('-', '').strip()
            logger.info(f'Parser: \n{res}')
            return res
        tree_struct = '\n'.join(get_yaml(response, 'response'))
        prompt = CODE_PARSING_SCHEMA_TEMPLATE.format(response_schema=tree_struct,
                                                     instruction=instruction, description=json_request['description'])
        messages = [{'role': "system", 'content': SYSTEM_PARSE}, {'role': 'user', 'content': prompt}]
        res = None
        status = False
        for i in range(0, 3):
            code = get_from_openai(model_name=self.model, temp=1,messages=messages)['content']
            if '```' in code and code.index('```') != code.rindex('```'):
                code = code[code.index('```'):code.rindex('```')]
                code = code.replace('```', '').replace('python', '').strip()
            elif '```' in code:
                code = code[code.index('```'):]
                code = code.replace('```', '').replace('python', '').strip()
            logger.info(f"Code: \n{code}")
            repl = PythonREPL(_globals={"response": response if type(response) == dict else json.loads(response)})
            status, res = repl.run(code)
            if status:
                break
            messages.append({'role': 'assistant', 'content': code})
            messages.append({'role': 'user',
                             'content': f'Your code encountered an error (bug) during runtime, and the specific error message is as follows:\n{res}\n\n'
                                        f'Please fix your bug and give me correct code to complete the instruction: {instruction}.\n'
                                        f'Just give me the code without any extra words\n'
                                        f'```python\n'
                                        f'[Python code]\n'
                                        f'```\n'})
            logger.info(f'Parser error: \n{res}')

        if res == None or 'None' in res or res == [] or '[]' in res or status == False or res == '':
            extract_prompt = LLM_PARSING_TEMPLATE.format(request=json_request, instruction=instruction,
                                                         response=json.dumps(response)[:1500])
            res = get_from_openai(model_name=self.model, temp=0,messages=[{"role": "user", 'content': extract_prompt}])['content']
        if len(res.split(' ')) > 1000:
            res = res[:1000] + '...'
        logger.info(f'Parser: \n{res}')
        self.add_traj(res)
        return res
