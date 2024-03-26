from utilize.apis import get_from_openai
from models.base import Base

icl_examples = {
    'tmdb': """"Example 1:
User Query: give me the number of movies directed by Sofia Coppola
Thought: First, use "https://api.themoviedb.org/3/search/person" to search for the **person id** of "Sofia Coppola"
API Selection: [1] https://api.themoviedb.org/3/search/person
Execution Result: the id of Sofia Coppola is **1769**
Thought: Continue, use "https://api.themoviedb.org/3/person/{person_id}/movie_credits" to get the number of **movies** directed by Sofia Coppola (person id is 1769)
API Selection: [2] https://api.themoviedb.org/3/person/{person_id}/movie_credits 
Execution Result: The movies directed by this person are Lost in Translation (153), The Virgin Suicides (1443), Marie Antoinette (1887), Somewhere (39210), Lick the Star (92657), The Bling Ring (96936), A Very Murray Christmas (364067), Bed, Bath and Beyond (384947), The Beguiled (399019)
Thought: Last, I am finished executing a plan and have the information the user asked.
Final Answer: Sofia Coppola has directed 9 movies

Example 2:
User Query: give me a image for the collection Star Wars
Thought: First, use "https://api.themoviedb.org/3/search/collection" to search for the collection Star Wars
API Selection: [5] https://api.themoviedb.org/3/search/collection
Execution Result: The **person id** of the Star Wars collection is 10
Thought: Continue, use "https://api.themoviedb.org/3/collection/{person_id}/images" to get the image of the Star Wars collection (10)
API Selection: [6] https://api.themoviedb.org/3/collection/{collection_id}/images
Execution Result:  The images url of Star Wars is abufowb4ui2bion2b2obo.jpg
Thought: I am finished executing a plan and have the information the user asked.
Final Answer: The images url of Star Wars is abufowb4ui2bion2b2obo.jpg

Example 3:
User Query: Who directed the top-1 rated movie?
Thought: First, use "https://api.themoviedb.org/3/movie/top_rated" to search the movie **id** for the top-1 rated movie
API Selection: [3] https://api.themoviedb.org/3/movie/top_rated
Execution Result: The name of the top-1 rated movie is The Godfather and the **movie id** is **238**
Thought: Continue, based on the movie id The Godfather (238), use "https://api.themoviedb.org/3/movie/{movie_id}/credits" to search for the **person id** of its director
API Selection: [4] https://api.themoviedb.org/3/movie/{movie_id}/credits
Execution Result: The name and id of the director of the movie The Godfather (**person id** 238) is Francis Ford Coppola (**person id** 1776)
Thought: Last, I am finished executing a plan and have the information the user asked.
Final Answer: Francis Ford Coppola directed the top-1 rated movie The Godfather""",

    "tmdb1": """Example 1:
User query: give me some movies performed by Tony Leung.
Thought: First, search person with name "Tony Leung"
API Selection: 
Executed results: Tony Leung's person_id is 1337
Thought: Continue, collect the list of movies performed by Tony Leung whose person_id is 1337
API Selection: 
Executed results: Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Tony Leung has performed in Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero

Example 2:
User query: Who wrote the screenplay for the most famous movie directed by Martin Scorsese?
Thought: First, search for the most popular movie directed by Martin Scorsese
API Selection: 
Executed results: Successfully called GET /search/person to search for the director "Martin Scorsese". The id of Martin Scorsese is 1032
Thought: Continue, search for the most popular movie directed by Martin Scorsese (1032)
API Selection: 
Executed results: Successfully called GET /person/{{person_id}}/movie_credits to get the most popular movie directed by Martin Scorsese. The most popular movie directed by Martin Scorsese is Shutter Island (11324)
Thought: Continue, search for the screenwriter of Shutter Island
API Selection: 
Executed results: The screenwriter of Shutter Island is Laeta Kalogridis (20294)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Laeta Kalogridis wrote the screenplay for the most famous movie directed by Martin Scorsese.
""",
    "spotify": """Example 1:
User query: set the volume to 20 and skip to the next track.
Plan step 1: set the volume to 20
API response: Successfully called PUT /me/player/volume to set the volume to 20.
Plan step 2: skip to the next track
API response: Successfully called POST /me/player/next to skip to the next track.
Thought: I am finished executing a plan and completed the user's instructions
Final Answer: I have set the volume to 20 and skipped to the next track.

Example 2:
User query: Make a new playlist called "Love Coldplay" containing the most popular songs by Coldplay
Plan step 1: search for the most popular songs by Coldplay
API response: Successfully called GET /search to search for the artist Coldplay. The id of Coldplay is 4gzpq5DPGxSnKTe4SA8HAU
Plan step 2: Continue. search for the most popular songs by Coldplay (4gzpq5DPGxSnKTe4SA8HAU)
API response: Successfully called GET /artists/4gzpq5DPGxSnKTe4SA8HAU/top-tracks to get the most popular songs by Coldplay. The most popular songs by Coldplay are Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b).
Plan step 3: make a playlist called "Love Coldplay"
API response: Successfully called GET /me to get the user id. The user id is xxxxxxxxx.
Plan step 4: Continue. make a playlist called "Love Coldplay"
API response: Successfully called POST /users/xxxxxxxxx/playlists to make a playlist called "Love Coldplay". The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Plan step 5: Add the most popular songs by Coldplay, Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b), to playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T)
API response: Successfully called POST /playlists/7LjHVU3t3fcxj5aiPFEW4T/tracks to add Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b) in playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T). The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Thought: I am finished executing a plan and have the data the used asked to create
Final Answer: I have made a new playlist called "Love Coldplay" containing Yellow and Viva La Vida by Coldplay.
"""
}


SYSTEM_PLAN = """You are an agent that access external APIs and plans solution to user queries by selecting appreciate APIs in a logic order. 
I will provide your external APIs to answer the user's query, and each API is called via HTTP request. You should make a plan to use appreciative APIs by iterating three steps: Thought, API Selection, and Execution Result. 
- Thought: In Thought step, you should reason current situation and specify which API to use. The Thought should be as specific as possible. It is better not to use pronouns in the plan, but to use the corresponding results obtained previously. For example, instead of "Get the most popular movie directed by this person", you should output "Get the most popular movie directed by Martin Scorsese (1032)". The Thought should be straightforward. If you want to search, sort, or filter, you can put the condition in your plan. For example, if the query is "Who is the lead actor of In the Mood for Love (person id 843)", instead of "Get the list of actors of In the Mood for Love", you should output "Get the lead actor of In the Mood for Love (843)".
- API Selection: based on the Thought, the API Selection step selects a corresponding API from the following API list. You should just select the API from the below list and DO NOT replace any parameter in the API.  
- Execution Result: your expected execution results for the selected API.
Each step must writen in one LINE without '\n'! During the Thought and Execution Result steps, you can mark the key information, e.g., movie id and person id with **. And when finishing, your plan should end with the `Final Answer' to give the final output to the user. 

Here are some examples: 

[1] /search/person
## description: Search for people.
[2] /person/{{person_id}}/movie_credits
## description: Get the movie credits for a person, the results contains various information such as popularity and release date.
[3] /movie/top_rated
## description: Get the top rated movies on TMDb.
[4] /movie/{{movie_id}}/credits
## description: Get the cast and crew for a movie.
[5] /search/collection
## description: Search for collections.
[6] /collection/{{collection_id}}/images
## description: Get the images for a collection by id.

Example 1:
User Query: give me the number of movies directed by Sofia Coppola
Thought: First, use "https://api.themoviedb.org/3/search/person" to search for the **person id** of "Sofia Coppola"
API Selection: [1] https://api.themoviedb.org/3/search/person
Execution Result: the id of Sofia Coppola is **1769**
Thought: Continue, use "https://api.themoviedb.org/3/person/{person_id}/movie_credits" to get the number of **movies** directed by Sofia Coppola (person id is 1769)
API Selection: [2] https://api.themoviedb.org/3/person/{person_id}/movie_credits 
Execution Result: The movies directed by this person are Lost in Translation (153), The Virgin Suicides (1443), Marie Antoinette (1887), Somewhere (39210), Lick the Star (92657), The Bling Ring (96936), A Very Murray Christmas (364067), Bed, Bath and Beyond (384947), The Beguiled (399019)
Thought: Last, I am finished executing a plan and have the information the user asked.
Final Answer: Sofia Coppola has directed 9 movies

Example 2:
User Query: give me a image for the collection Star Wars
Thought: First, use "https://api.themoviedb.org/3/search/collection" to search for the collection Star Wars
API Selection: [5] https://api.themoviedb.org/3/search/collection
Execution Result: The **person id** of the Star Wars collection is 10
Thought: Continue, use "https://api.themoviedb.org/3/collection/{person_id}/images" to get the image of the Star Wars collection (10)
API Selection: [6] https://api.themoviedb.org/3/collection/{collection_id}/images
Execution Result:  The images url of Star Wars is abufowb4ui2bion2b2obo.jpg
Thought: I am finished executing a plan and have the information the user asked.
Final Answer: The images url of Star Wars is abufowb4ui2bion2b2obo.jpg

Example 3:
User Query: Who directed the top-1 rated movie?
Thought: First, use "https://api.themoviedb.org/3/movie/top_rated" to search the movie **id** for the top-1 rated movie
API Selection: [3] https://api.themoviedb.org/3/movie/top_rated
Execution Result: The name of the top-1 rated movie is The Godfather and the **movie id** is **238**
Thought: Continue, based on the movie id The Godfather (238), use "https://api.themoviedb.org/3/movie/{movie_id}/credits" to search for the **person id** of its director
API Selection: [4] https://api.themoviedb.org/3/movie/{movie_id}/credits
Execution Result: The name and id of the director of the movie The Godfather (**person id** 238) is Francis Ford Coppola (**person id** 1776)
Thought: Last, I am finished executing a plan and have the information the user asked.
Final Answer: Francis Ford Coppola directed the top-1 rated movie The Godfather"""

PLANNER_PROMPT="""Using the following API to the user query. You can **only** use the API listed here.

Starting below, you should follow this format:
User query: the query a User wants help with related to the API.
Thought: the first step of your plan for how to solve the query. Your must specify the key information you have got (e.g., person id and movie id) and which API to use.
API Selection: selecting an API from the above list based on your `Thought`. Just select an API from the above list, do not change the name of the API.
Execution Result: the expected result after executing the API. You should specify what you want to obtain from the API request.
Thought: Thought: based on the API response, the second step of your plan for how to solve the query.  Pay attention to the specific API called in the last step API response. If a proper API is called, then the response may be wrong and you should give a new plan.
... (The three steps: Thought, API Selection, and  Execution Result, can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for.
Final Answer: the final output from executing the plan

Begin!
API List:
{api_list} 

User query: {query}
{hidden}"""

class GroAgent(Base):

    def __init__(self, model='gpt-3.5-turbo', endpoints=None,role='PlanGPT',url=None):
        super().__init__(model=model, endpoints=endpoints,role=role,url=url)
        self.tools = []
        self.endpoints = endpoints
        self.tokens=[]

    def generate(self, query, hidden, api_type):
        icl_example = icl_examples[api_type]
        api_list = []
        for i, tool in enumerate(self.endpoints, start=1):
            description = tool["description"].replace('\n','')
            api_list.append(f'[{i}] {tool["url"]}\n## description: {description}')
        api_list = '\n'.join(api_list)

        prompt = PLANNER_PROMPT.format(icl_example=icl_example, query=query, api_list=api_list, hidden=hidden)
        res=''
        for i in range(0,4):
            res = get_from_openai(model_name=self.model, temp=0, max_len=1000, stop=['Execution Result'],
                                  messages=[{'role': "system", 'content': SYSTEM_PLAN},
                                            {'role': 'user', 'content': prompt}])['content']
            if 'final answer' in res.lower():
                res = res.lower()
                id1 = res.index('final answer')
                return 'Finish',res[id1:]
            if 'API Selection:' in res:
                thought, action = res.split('API Selection:')
                thought = self.normalize(thought)
                action = self.normalize(action)
                return thought,action

        res=[e for e in res.split('\n') if e.strip()!='']
        thought = '\n'.join(res)
        thought=self.normalize(thought)
        prompt += thought + '\nAPI Selection: '
        action = get_from_openai(model_name=self.model, temp=0, max_len=500, stop='Thought: ',
                                messages=[{'role': "system", 'content': SYSTEM_PLAN},
                                          {'role': 'user', 'content': prompt}])['content']
        action=self.normalize(action)

        return self.normalize(thought),self.normalize(action)
