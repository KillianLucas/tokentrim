from tokentrim import trim

def test_trim_short_message():
    messages = [{'role': 'user', 'content': 'Hello!'}]
    model = 'gpt-3.5-turbo'
    trimmed = trim(messages, model)
    assert trimmed == messages

def test_trim_long_message():
    long_message = 'A' * 100000
    messages = [{'role': 'user', 'content': long_message}]
    model = 'gpt-3.5-turbo'
    trimmed = trim(messages, model)
    assert len(trimmed[0]['content']) < len(long_message)

def test_trim_preserves_system_message():
    system_message = 'Hello, user!'
    messages = [{'role': 'user', 'content': 'A' * 100000}]
    model = 'gpt-3.5-turbo'
    trimmed = trim(messages, model, system_message=system_message)
    assert trimmed[0]['content'] == system_message

def test_trim_invalid_model():
    messages = [{'role': 'user', 'content': 'Hello!'}]
    model = 'invalid-model'
    try:
        trim(messages, model)
    except ValueError:
        pass
    else:
        assert False, "Expected a ValueError to be raised for invalid model"