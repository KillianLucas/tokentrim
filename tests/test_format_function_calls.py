from tokentrim.format_function_calls import format_tool


def test_simple_tool():
  simple_tool = dict(
    name="get_location",
    description="Get the user's location",
    parameters={
      "type": "object",
      "properties": {}
    },
  )
  actual = format_tool(simple_tool)
  expected = """// Get the user's location
type get_location = () => any;

"""
  assert actual == expected


def test_create_model_tool():
  create_model_tool = dict(
    name="create_model",
    description="Create a new conversational agent",
    parameters={
      "title": "GptParams",
      "type": "object",
      "properties": {
        "temperature": {
          "title": "Temperature",
          "default": 1,
          "minimum": 0,
          "maximum": 2,
          "type": "number",
        },
        "max_tokens": {
          "title": "Max Tokens",
          "description": "The maximum response length of the model",
          "default": 512,
          "type": "integer",
        },
        "reserve_tokens": {
          "title": "Reserve Tokens",
          "description": "Number of tokens reserved for the model's output",
          "default": 512,
          "type": "integer",
        },
      },
      "additionalProperties": False,
    },
  )
  actual = format_tool(create_model_tool)
  expected = """// Create a new conversational agent
type create_model = (_: {
temperature?: number, // default: 1.0
// The maximum response length of the model
max_tokens?: number, // default: 512
// Number of tokens reserved for the model's output
reserve_tokens?: number, // default: 512
}) => any;

"""
  assert actual == expected


def test_send_message_tool():
  send_message_tool = dict(
    name="send_message",
    description="Send a new message",
    parameters={
      "title": "ConversationCreate",
      "type": "object",
      "properties": {
        "params": {
          "$ref": "#/definitions/ConversationParams"
        },
        "messages": {
          "title": "Messages",
          "type": "array",
          "items": {
            "$ref": "#/definitions/MessageCreate"
          },
        },
      },
      "required": ["params", "messages"],
      "definitions": {
        "ConversationParams": {
          "title": "ConversationParams",
          "description":
          "Parameters to use for the conversation. Extra fields are permitted and\npassed to the model directly.",
          "type": "object",
          "properties": {
            "model": {
              "title": "Model",
              "description": "Completion model to use for the conversation",
              "pattern": "^.+:.+$",
              "example": "openai:Gpt35Model",
              "type": "string",
            },
            "model_params": {
              "title": "Model Params",
              "type": "object",
              "properties": {},
              "additionalProperties": True,
            },
            "features": {
              "title": "Features",
              "description":
              "Set of enabled features for this conversation and the parameters for them.",
              "example": {
                "test:dummy_feature": {
                  "enable_weather": True
                }
              },
              "type": "object",
            },
          },
          "required": ["model", "model_params", "features"],
        },
        "MessageRole": {
          "title": "MessageRole",
          "description": "An enumeration.",
          "enum": ["user", "assistant", "system", "tool_call", "tool_result"],
          "type": "string",
        },
        "MessageCreate": {
          "title": "MessageCreate",
          "type": "object",
          "properties": {
            "role": {
              "$ref": "#/definitions/MessageRole"
            },
            "name": {
              "title": "Name",
              "description":
              "\n            Sender of the message. For tool_call and tool_result, this is the\n            name of the tool being referenced. Otherwise, it is optional.\n            ",
              "example": "user",
              "type": "string",
            },
            "content": {
              "title": "Content",
              "description":
              "\n            Arbitrary data. For regular messages (not tool calls/results), this\n            must include a 'text' field containing the message text.\n            ",
              "example": {
                "text": "Why is the sky blue?"
              },
              "additionalProperties": True,
              "type": "object",
            },
          },
          "required": ["role", "content"],
        },
      },
    },
  )
  actual = format_tool(send_message_tool)
  expected = """// Send a new message
type send_message = (_: {
// Parameters to use for the conversation. Extra fields are permitted and
// passed to the model directly.
params: {
  model: string,
  model_params: object,
},
messages: {
  role: "user" | "assistant" | "system" | "tool_call" | "tool_result",
  name?: string,
  content: object,
}[],
}) => any;

"""
  assert actual == expected
