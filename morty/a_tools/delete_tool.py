from memgpt.client.client import create_client


client = create_client()
client.delete_tool(name="justins_function")
client.delete_tool(name="justins_function2")
# client.delete_tool(name="roll_special_dice")
# client.delete_tool(name="get_secret")
