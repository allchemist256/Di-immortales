import copy
import json
import os
import random
import time
import uuid
import warnings
from typing import List, Optional, Union

import requests

from memgpt.constants import CLI_WARNING_PREFIX, JSON_ENSURE_ASCII
from memgpt.credentials import MemGPTCredentials
from memgpt.data_types import Message

from memgpt.local_llm.chat_completion_proxy import get_chat_completion
from memgpt.local_llm.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION,
)
from memgpt.models.chat_completion_request import (
    ChatCompletionRequest,
    Tool,
    cast_message_to_subtype,
)
from memgpt.models.chat_completion_response import ChatCompletionResponse
from memgpt.models.pydantic_models import LLMConfigModel, OptionState
from memgpt.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)

LLM_API_PROVIDER_OPTIONS = ["openai", "azure", "anthropic", "google_ai", "cohere", "local"]


# TODO update to use better types
def add_inner_thoughts_to_functions(
    functions: List[dict],
    inner_thoughts_key: str,
    inner_thoughts_description: str,
    inner_thoughts_required: bool = True,
    # inner_thoughts_to_front: bool = True,  TODO support sorting somewhere, probably in the to_dict?
) -> List[dict]:
    """Add an inner_thoughts kwarg to every function in the provided list"""
    # return copies
    new_functions = []

    # functions is a list of dicts in the OpenAI schema (https://platform.openai.com/docs/api-reference/chat/create)
    for function_object in functions:
        function_params = function_object["parameters"]["properties"]
        required_params = list(function_object["parameters"]["required"])

        # if the inner thoughts arg doesn't exist, add it
        if inner_thoughts_key not in function_params:
            function_params[inner_thoughts_key] = {
                "type": "string",
                "description": inner_thoughts_description,
            }

        # make sure it's tagged as required
        new_function_object = copy.deepcopy(function_object)
        if inner_thoughts_required and inner_thoughts_key not in required_params:
            required_params.append(inner_thoughts_key)
            new_function_object["parameters"]["required"] = required_params

        new_functions.append(new_function_object)

    # return a list of copies
    return new_functions


def unpack_inner_thoughts_from_kwargs(
    response: ChatCompletionResponse,
    inner_thoughts_key: str,
) -> ChatCompletionResponse:
    """Strip the inner thoughts out of the tool call and put it in the message content"""
    if len(response.choices) == 0:
        raise ValueError(f"Unpacking inner thoughts from empty response not supported")

    new_choices = []
    for choice in response.choices:
        msg = choice.message
        if msg.role == "assistant" and msg.tool_calls and len(msg.tool_calls) >= 1:
            if len(msg.tool_calls) > 1:
                warnings.warn(f"Unpacking inner thoughts from more than one tool call ({len(msg.tool_calls)}) is not supported")
            # TODO support multiple tool calls
            tool_call = msg.tool_calls[0]

            try:
                # Sadly we need to parse the JSON since args are in string format
                func_args = dict(json.loads(tool_call.function.arguments))
                if inner_thoughts_key in func_args:
                    # extract the inner thoughts
                    inner_thoughts = func_args.pop(inner_thoughts_key)

                    # replace the kwargs
                    new_choice = choice.model_copy(deep=True)
                    new_choice.message.tool_calls[0].function.arguments = json.dumps(func_args, ensure_ascii=JSON_ENSURE_ASCII)
                    # also replace the message content
                    if new_choice.message.content is not None:
                        warnings.warn(f"Overwriting existing inner monologue ({new_choice.message.content}) with kwarg ({inner_thoughts})")
                    new_choice.message.content = inner_thoughts

                    # save copy
                    new_choices.append(new_choice)
                else:
                    warnings.warn(f"Did not find inner thoughts in tool call: {str(tool_call)}")

            except json.JSONDecodeError as e:
                warnings.warn(f"Failed to strip inner thoughts from kwargs: {e}")
                raise e

    # return an updated copy
    new_response = response.model_copy(deep=True)
    new_response.choices = new_choices
    return new_response


def is_context_overflow_error(exception: requests.exceptions.RequestException) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages)"""
    from memgpt.utils import printd

    match_string = "maximum context length"

    # Backwards compatibility with openai python package/client v0.28 (pre-v1 client migration)
    if match_string in str(exception):
        printd(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    # Based on python requests + OpenAI REST API (/v1)
    elif isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None and "application/json" in exception.response.headers.get("Content-Type", ""):
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    printd(f"HTTPError occurred, but couldn't find error field: {error_details}")
                    return False
                else:
                    error_details = error_details["error"]

                # Check for the specific error code
                if error_details.get("code") == "context_length_exceeded":
                    printd(f"HTTPError occurred, caught error code {error_details.get('code')}")
                    return True
                # Soft-check for "maximum context length" inside of the message
                elif error_details.get("message") and "maximum context length" in error_details.get("message"):
                    printd(f"HTTPError occurred, found '{match_string}' in error message contents ({error_details})")
                    return True
                else:
                    printd(f"HTTPError occurred, but unknown error message: {error_details}")
                    return False
            except ValueError:
                # JSON decoding failed
                printd(f"HTTPError occurred ({exception}), but no JSON error message.")

    # Generic fail
    else:
        return False


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    # List of OpenAI error codes: https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_client.py#L227-L250
    # 429 = rate limit
    error_codes: tuple = (429,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        pass

        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            except requests.exceptions.HTTPError as http_err:
                # Retry on specified errors
                if http_err.response.status_code in error_codes:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    # printd(f"Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying...")
                    print(
                        f"{CLI_WARNING_PREFIX}Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying..."
                    )
                    time.sleep(delay)
                else:
                    # For other HTTP errors, re-raise the exception
                    raise

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def create(
    # agent_state: AgentState,
    llm_config: LLMConfigModel,
    messages: List[Message],
    user_id: uuid.UUID = None,  # option UUID to associate request with
    functions: list = None,
    functions_python: list = None,
    function_call: str = "auto",
    # hint
    first_message: bool = False,
    # use tool naming?
    # if false, will use deprecated 'functions' style
    use_tool_naming: bool = True,
    # streaming?
    stream: bool = False,
    stream_inferface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    # TODO move to llm_config?
    # if unspecified (None), default to something we've tested
    inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
) -> ChatCompletionResponse:
    """Return response to chat completion with backoff"""
    from memgpt.utils import printd

    printd(f"Using model {llm_config.model_endpoint_type}, endpoint: {llm_config.model_endpoint}")

    # TODO eventually refactor so that credentials are passed through

    credentials = MemGPTCredentials.load()

    if function_call and not functions:
        printd("unsetting function_call because functions is None")
        function_call = None

    # print("HELLO")

    # openai

    if stream:
        raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
    return get_chat_completion(
        model=llm_config.model,
        messages=messages,
        functions=functions,
        functions_python=functions_python,
        function_call=function_call,
        context_window=llm_config.context_window,
        endpoint=llm_config.model_endpoint,
        endpoint_type=llm_config.model_endpoint_type,
        wrapper=llm_config.model_wrapper,
        user=str(user_id),
        # hint
        first_message=first_message,
        # auth-related
        auth_type=credentials.openllm_auth_type,
        auth_key=credentials.openllm_key,
    )
