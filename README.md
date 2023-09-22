# TokenTrim

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

TokenTrim intelligently trims OpenAI `messages` to fit within a model's token limit (shortening a message by removing characters from the middle), making it easy to avoid exceeding the maximum token count.

It's best suited for use directly in OpenAI API calls:

```python
import tokentrim as tt

model = "gpt-4"

response = openai.ChatCompletion.create(
  model=model,
  messages=tt.trim(messages, model) # Trims old messages to fit under model's max token count
)
```

TokenTrim's behavior is based on OpenAI's own [best practices.](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)

## Installation

Use the package manager pip to install TokenTrim:

```bash
pip install tokentrim
```

## Usage

The primary function in the TokenTrim library is `trim()`. This function receives a list of messages and a model name, and it returns a trimmed list of messages that should be within the model's token limit.

```python
import tokentrim as tt

# Force a system_message to be prepended to your messages list. This will not be trimmed.
system_message = "You are a helpful assistant."

response = openai.ChatCompletion.create(
  model=model,
  messages=tt.trim(messages, model, system_message=system_message)
)
```

### Parameters

- `messages` : A list of message objects to be trimmed. Each message is a dictionary with 'role' and 'content'.
- `model` : The OpenAI model being used (e.g., 'gpt-4', 'gpt-4-32k'). This determines the token limit.
- `system_message` (optional): A system message to preserve at the start of the conversation.
- `trim_ratio` (optional): Target ratio of tokens to use after trimming. Default is 0.75, meaning it will trim messages so they use about 75% of the model's token limit.
- `return_response_tokens` (optional): If set to True, the function also returns the number of tokens left available for the response after trimming.

### Return Value

By default, `trim()` returns the trimmed list of messages. If `return_response_tokens` is set to True, it returns a tuple where the first element is the trimmed list of messages, and the second element is the number of tokens left available for the response.

## License

This project is licensed under the terms of the MIT license.
