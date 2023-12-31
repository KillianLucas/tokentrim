# HACK: this is adapted from: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# We should probably collaborate on putting this into some sort of shared package that can be imported
# by various projects that need this kind of info.
# I've reached out to the [AI Engineer Foundation](https://github.com/AI-Engineer-Foundation) about creating
# and maintaining something like this as a single source of truth that everyone can import, extend, and benefit from.

MODEL_MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-16k-0613": 16385,
    "text-davinci-003": 4097,
    "text-curie-001": 2049,
    "text-babbage-001": 2049,
    "text-ada-001": 2049,
    "babbage-002": 16384,
    "davinci-002": 16384,
    "gpt-3.5-turbo-instruct": 8192,
    "claude-instant-1": 100000,
    "claude-instant-1.2": 100000,
    "claude-2": 100000,
    "text-bison": 8192,
    "text-bison@001": 8192,
    "chat-bison": 4096,
    "chat-bison@001": 4096,
    "chat-bison-32k": 32000,
    "code-bison": 6144,
    "code-bison@001": 6144,
    "code-gecko@001": 2048,
    "code-gecko@latest": 2048,
    "codechat-bison": 6144,
    "codechat-bison@001": 6144,
    "codechat-bison-32k": 32000,
    "command-nightly": 4096,
    "command": 4096,
    "command-light": 4096,
    "command-medium-beta": 4096,
    "command-xlarge-beta": 4096,
    "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1": 4096,
    "openrouter/openai/gpt-3.5-turbo": 4095,
    "openrouter/openai/gpt-3.5-turbo-16k": 16383,
    "openrouter/openai/gpt-4": 8192,
    "openrouter/anthropic/claude-instant-v1": 100000,
    "openrouter/anthropic/claude-2": 100000,
    "openrouter/google/palm-2-chat-bison": 8000,
    "openrouter/google/palm-2-codechat-bison": 8000,
    "openrouter/meta-llama/llama-2-13b-chat": 4096,
    "openrouter/meta-llama/llama-2-70b-chat": 4096,
    "openrouter/meta-llama/codellama-34b-instruct": 8096,
    "openrouter/nousresearch/nous-hermes-llama2-13b": 4096,
    "openrouter/mancer/weaver": 8000,
    "openrouter/gryphe/mythomax-l2-13b": 8192,
    "openrouter/jondurbin/airoboros-l2-70b-2.1": 4096,
    "openrouter/undi95/remm-slerp-l2-13b": 6144,
    "openrouter/pygmalionai/mythalion-13b": 4096,
    "openrouter/mistralai/mistral-7b-instruct": 4096,
    "j2-ultra": 8192,
    "j2-mid": 8192,
    "j2-light": 8192,
    "dolphin": 4096,
    "chatdolphin": 4096,
    "luminous-base": 2048, 
    "luminous-base-control": 2048, 
    "luminous-extended": 2048, 
    "luminous-extended-control": 2048, 
    "luminous-supreme": 2048, 
    "luminous-supreme-control": 2048, 
    "ai21.j2-mid-v1": 8191, 
    "ai21.j2-ultra-v1": 8191, 
    "amazon.titan-text-lite-v1": 8000, 
    "amazon.titan-text-express-v1": 8000, 
    "anthropic.claude-v1": 100000, 
    "anthropic.claude-v2": 100000, 
    "anthropic.claude-instant-v1": 100000, 
    "cohere.command-text-v14": 4096, 
    "together-ai-up-to-3b": 1000,
    "ollama/llama2:13b": 4096,
    "ollama/llama2:70b": 4096,
    "ollama/llama2-uncensored": 4096,
    "deepinfra/meta-llama/Llama-2-70b-chat-hf": 6144,
    "deepinfra/codellama/CodeLlama-34b-Instruct-hf": 4096,
    "deepinfra/meta-llama/Llama-2-13b-chat-hf": 4096,
    "deepinfra/meta-llama/Llama-2-7b-chat-hf": 4096,
    "deepinfra/mistralai/Mistral-7B-Instruct-v0.1": 4096,
    "deepinfra/jondurbin/airoboros-l2-70b-gpt4-1.4.1": 4096,
}