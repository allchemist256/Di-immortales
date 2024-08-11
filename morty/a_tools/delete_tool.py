from memgpt.client.client import create_client
from memgpt.config import MemGPTConfig
from memgpt.constants import USER_ID
from memgpt.metadata import MetadataStore

config = MemGPTConfig.load()
ms = MetadataStore(config)
client = create_client()
# print(USER_ID)
# print(ms.list_personas(USER_ID))
ms.delete_persona("sam", USER_ID)
# client.delete_persona("sam")
# for persona in ms.list_personas():
# print(persona)

# client = create_client()
# client.delete_tool(name="justins_function")
# client.delete_tool(name="justins_function2")
# client.delete_tool(name="roll_special_dice")
# client.delete_tool(name="get_secret")
