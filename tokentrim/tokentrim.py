import tiktoken
from typing import List, Dict, Any, Tuple, Optional

MODEL_MAX_TOKENS = {
  'gpt-4': 8192,
  'gpt-4-0613': 8192,
  'gpt-4-32k': 32768,
  'gpt-4-32k-0613': 32768,
  'gpt-3.5-turbo': 4096,
  'gpt-3.5-turbo-16k': 16384,
  'gpt-3.5-turbo-0613': 4096,
  'gpt-3.5-turbo-16k-0613': 16384,
}


def num_tokens_from_messages(messages: List[Dict[str, Any]],
                             model: str) -> int:
  """
  Function to return the number of tokens used by a list of messages.
  """
  # Attempt to get the encoding for the specified model
  try:
    encoding = tiktoken.encoding_for_model(model)
  except KeyError:
    print("Warning: model not found. Using cl100k_base encoding.")
    encoding = tiktoken.get_encoding("cl100k_base")

  # Token handling specifics for different model types
  if model in {
      "gpt-3.5-turbo-0613",
      "gpt-3.5-turbo-16k-0613",
      "gpt-4-0314",
      "gpt-4-32k-0314",
      "gpt-4-0613",
      "gpt-4-32k-0613",
  }:
    tokens_per_message = 3
    tokens_per_name = 1
  elif model == "gpt-3.5-turbo-0301":
    tokens_per_message = 4
    tokens_per_name = -1
  elif "gpt-3.5-turbo" in model:
    return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
  elif "gpt-4" in model:
    return num_tokens_from_messages(messages, model="gpt-4-0613")
  else:
    raise NotImplementedError(
      f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
    )

  # Calculate the number of tokens
  num_tokens = 0
  for message in messages:
    num_tokens += tokens_per_message
    for key, value in message.items():
      num_tokens += len(encoding.encode(value))
      if key == "name":
        num_tokens += tokens_per_name

  num_tokens += 3
  return num_tokens


def shorten_message_to_fit_limit(message: Dict[str, Any], tokens_needed: int,
                                 model: str) -> None:
  """
  Shorten a message to fit within a token limit by removing characters from the middle.
  """

  content = message["content"]
  left_half = content[:len(content) // 2]
  right_half = content[len(content) // 2:]

  while True:
    trimmed_content = left_half + '...' + right_half
    message["content"] = trimmed_content
    if num_tokens_from_messages([message], model) > tokens_needed:
      if len(left_half) > len(right_half):
        # Cut from the left half if it's longer or equal
        left_half = left_half[:-1]
      else:
        # Otherwise cut from the right half
        right_half = right_half[1:]
    else:
      break


def trim(
  messages: List[Dict[str, Any]],
  model: str,
  system_message: Optional[str] = None,
  trim_ratio: float = 0.75,
  return_response_tokens: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
  """
    Trim a list of messages to fit within a model's token limit.

    Args:
        messages: Input messages to be trimmed. Each message is a dictionary with 'role' and 'content'.
        model: The OpenAI model being used (determines the token limit).
        system_message: Optional system message to preserve at the start of the conversation.
        trim_ratio: Target ratio of tokens to use after trimming. Default is 0.75, meaning it will trim messages so they use about 75% of the model's token limit.
        return_response_tokens: If True, also return the number of tokens left available for the response after trimming.

    Returns:
        Trimmed messages and optionally the number of tokens available for response.
    """

  # Initialize max_tokens
  max_tokens = int(MODEL_MAX_TOKENS[model] * trim_ratio)

  # Deduct the system message tokens from the max_tokens if system message exists
  if system_message:
    system_message_event = {"role": "system", "content": system_message}
    system_message_tokens = num_tokens_from_messages([system_message_event],
                                                     model)
    max_tokens -= system_message_tokens

    if system_message_tokens > max_tokens:
      raise ValueError("System message exceeds token limit")

  final_messages = []

  # Reverse the messages so we process oldest messages first
  messages = messages[::-1]

  # Process the messages
  for message in messages:
    temp_messages = [message] + final_messages

    if num_tokens_from_messages(temp_messages, model) <= max_tokens:
      # If adding the next message doesn't exceed the token limit, add it to final_messages
      final_messages = [message] + final_messages
    else:
      # If adding the next message exceeds the token limit, try trimming it
      shorten_message_to_fit_limit(
        message, max_tokens - num_tokens_from_messages(final_messages, model),
        model)

      # If the trimmed message can fit, add it
      if num_tokens_from_messages([message] + final_messages,
                                  model) <= max_tokens:
        final_messages = [message] + final_messages

      # Regardless if the trimmed message fits or not, break
      break

  # Add system message to the start of final_messages if it exists
  if system_message:
    final_messages = [system_message_event] + final_messages

  if return_response_tokens:
    response_tokens = max_tokens - num_tokens_from_messages(
      final_messages, model)
    return final_messages, response_tokens

  return final_messages
