import json
import requests
import subprocess

URL = "http://localhost:10086/v1/chat/completions"

def ExecuteJSCode(code):
    try:
        Result = subprocess.run(
            ["node", "-e", code], 
            capture_output = True, text = True, timeout = 5
        )
        if Result.returncode != 0:
            return f"Execution Error: {Result.stderr}"
        return Result.stdout.strip()
    
    except Exception as e:
        return f"System Error: {str(e)}"

Messages = [{
    "role"    : "user",
    "content" : [{
        "type": "text",
        "text": (
            "任务1: 请用JavaScript计算斐波那契数列的第11项是多少?\n"
            "任务2: 请告诉我文件的名称和内容是什么?\n"
            "任务3: 请告诉我图片中有什么?"
        )
    }, {
        "type": "file",
        "file": {
            "file_data": "SSBhbSBsaXR0bGUgYmx1ZSB3aGFsZS4=",
            "filename" : "Who am I.txt"
        }
    }, {
        "type"     : "image_url",
        "image_url": {
            "url": r"\ReadmeSrc\Gamma 0.1.2 Architecture.svg"
        }
    }]
}]

Tools    = [{
    "type"    : "function",
    "function": {
        "name"       : "ExecJS",
        "description": "Execute JavaScript code. Ensure the result is printed.",
        "parameters" : {
            "type"       : "object",
            "properties" : {
                "JSCode" : { "type": "string", "description": "The JS code to execute." }
            },
            "required"   : ["JSCode"]
        }
    }
}]

print(f"{'=' * 20} Start Completions {'=' * 20}")

while True:
    Payload = {
        "model"      : "deepseek",
        "messages"   : Messages,
        "tools"      : Tools,
        "tool_choice": "auto"
    }

    Response = requests.post(URL, json = Payload)
    if Response.status_code != 200:
        print("API request failed:", Response.text)
        break

    ResponseBody = Response.json()
    Message      = ResponseBody["choices"][0]["message"]
    print(f"\nModel Reply: \n{Message.get('content', 'NO CONTENT')}")

    Messages.append(Message)
    ToolCalls = Message.get("tool_calls", [])

    if ToolCalls:
        print(f"\nTool Calls: \n{json.dumps(ToolCalls, indent=2)}")

        for ToolCall in ToolCalls:
            if ToolCall["function"]["name"] == "ExecJS":

                Arguments = json.loads(ToolCall["function"]["arguments"])
                Code      = Arguments["JSCode"]                
                Result    = ExecuteJSCode(Code)        
                print(f"Result: {Result}")
                        
                ToolMessage = {
                    "role"        : "tool",
                    "tool_call_id": ToolCall["id"],
                    "content"     : str(Result)
                }

                Messages.append(ToolMessage)

                del Message["tool_calls"]

    else:
        break
    