# from memgpt.agent import Agent
from memgpt import create_client
from memgpt.memory import ChatMemory
from memgpt.utils import get_human_text, list_human_files, list_persona_files

from memgpt.agent import Agent


def get_secret(self: Agent) -> str:
    """
    Get Secret String.

    Returns:
        string: The secret string

    Example:
        >>> get_secret()
        "this is a secret string"
    """
    print("hello")
    with open("/home/fury15/Di-immortales/morty/copy.txt", "w") as file:
        file.write("Your text goes here")

    return "output_string_161289"


# def roll_special_dice(self: Agent) -> str:
#     """
#     Simulate the roll of a roll_special_dice die.

#     Returns:
#         int: A random integer between two special numbers, representing the roll.

#     Example:
#         >>> roll_special_dice()
#         1824178213611  # This is an example output and may vary each time the function is called.
#     """
#     import random

#     output_string = f"You rolled a {1}"
#     return output_string


def main():
    #     # Create a `LocalClient` (you can also use a `RESTClient`, see the memgpt_rest_client.py example)
    client = create_client()
    #     # create tool
    #     # client.delete_tool(name="roll_d9")
    tool = client.create_tool(get_secret, update=True)

    #     # google search persona
    #     # Create an agent
    me = client.get_human("Justin").text
    morty = client.get_persona("morty").text
    agent_state = client.create_agent(
        name="tlt",
        memory=ChatMemory(human=me, persona=morty),
        # memory=ChatMemory(human="Justin", persona="morty"),
        metadata={"human:": "Justin", "persona": "morty"},
        tools=["get_secret"],
    )

    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")


#     # # # Send a message to the agent
#     # print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")
#     # send_message_response = client.user_message(
#     #     agent_id=agent_state.id, message="What is the weather in Berkeley?"
#     # )
#     # print(
#     #     f"Recieved response: \n{json.dumps(send_message_response.messages, indent=4)}"
#     # )

#     # # Delete agent
#     # client.delete_agent(agent_id=agent_state.id)
#     # print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")
#     tools = client.list_tools()

#     print(tools[-1].name)


if __name__ == "__main__":
    main()
