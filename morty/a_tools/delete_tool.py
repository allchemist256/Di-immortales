from memgpt.client.client import create_client
from memgpt.config import MemGPTConfig
from memgpt.constants import USER_ID
from memgpt.metadata import MetadataStore

config = MemGPTConfig.load()
ms = MetadataStore(config)
client = create_client()
# for agent in ms.list_agents(USER_ID):
#     # print(agent.name)
#     if agent.name != "Morty":
#         # print(agent.name)
#         ms.delete_agent(agent.id)

# print(USER_ID)
# print(ms.list_personas(USER_ID))
# ms.delete_human("cs_phd", USER_ID)
# ms.delete_human("basic", USER_ID)
# ms.delete_persona("memgpt_starter", USER_ID)
# ms.delete_persona("sam_simple_pov_gpt35", USER_ID)
# ms.delete_persona("google_search_persona", USER_ID)
# ms.delete_persona("memgpt_doc", USER_ID)
# ms.delete_persona("anna_pa", USER_ID)
# ms.delete_persona("sam_pov", USER_ID)

# client.delete_persona("sam")
# for persona in ms.list_personas():
# print(persona)

# client = create_client()
# client.delete_tool(name="justins_function")
# client.delete_tool(name="justins_function2")
# client.delete_tool(name="roll_special_dice")
# client.delete_tool(name="get_secret")
