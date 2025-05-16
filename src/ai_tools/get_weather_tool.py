import json

import requests

from src.common.open_api_utils import get_open_ai_client


# inspiration https://platform.openai.com/docs/guides/function-calling?api-mode=responses


def get_weather(latitude, longitude):
    # response = requests.get(
    #     f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    # data = response.json()
    # return data['current']['temperature_2m']
    return "Very nice weather"


#
# Step 1: Call model with get_weather tool defined
#
client = get_open_ai_client()

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]

input_messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]

response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)

#
# Step 2: Model decides to call function(s) – model returns the name and input arguments.
#
print(response.output)

#
# Step 3: Execute get_weather function
#
tool_call = response.output[0]
args = json.loads(tool_call.arguments)
result = get_weather(args["latitude"], args["longitude"])

#
# Step 4: Supply result and call model again
#
input_messages.append(tool_call)  # append model's function call message
input_messages.append({  # append result message
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

response_2 = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools
)
print(response_2.output_text)
