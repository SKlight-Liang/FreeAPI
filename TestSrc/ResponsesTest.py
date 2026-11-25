import json
import requests
import subprocess

URL = "http://localhost:10086/v1/responses"

def ExecuteJSCode(Object):
    try:
        Result = subprocess.run(
            ["node", "-e", Object["JSCode"]], 
            capture_output = True, text = True, timeout = 5
        )
        if Result.returncode != 0:
            return f"Execution Error: {Result.stderr}"
        return Result.stdout.strip()
    except Exception as e:
        return f"System Error: {str(e)}"

def ExecutePythonCode(Object):
    try:
        Result = subprocess.run(
            ["python", "-c", Object["PythonCode"]], 
            capture_output = True, text = True, timeout = 5
        )
        if Result.returncode != 0:
            return f"Execution Error: {Result.stderr}"
        return Result.stdout.strip()
    except Exception as e:
        return f"System Error: {str(e)}"

def GetWeather(Object):
    Location = Object["Location"]
    return f"{Location}的天气信息： 冷的不行，零下好几百度。"

ToolFunctions = {
    "ExecJS"   : ExecuteJSCode,
    "ExecPython": ExecutePythonCode,
    "GetWeather": GetWeather
}

InputItems = [{
    "role"   : "user",
    "content": [{
        "type": "input_text",
        "text": (
            "任务0：文件名称和内容分别是什么？\n"
            "任务1：请计算斐波那契数列的第11项(使用两种编程语言解决)。\n"
            "任务2：北京的天气怎么样？\n"
            "任务3：图片描述了什么？"
        )
    }, {
        "type"     : "input_file",
        "detail"   : "low",
        "file_data": "SSBhbSBsaXR0bGUgYmx1ZSB3aGFsZS4=",
        "file_name": "Who am I.txt"
    }, {
        "type"     : "input_image",
        "image_url": r"ReadmeSrc\Gamma 0.1.2 Architecture.svg"
    }]
}]

Tools = [{
    "type"       : "function",
    "name"       : "ExecJS",
    "description": "Execute JavaScript code. Ensure the result is printed.",
    "parameters" : {
        "type"       : "object",
        "properties" : {
            "JSCode" : { 
                "type"       : "string", 
                "description": "The JS code to execute." 
            }
        },
        "required"   : ["JSCode"]
    }
}, {
    "type"       : "function",
    "name"       : "ExecPython",
    "description": "Execute Python code. Pay attention to indentation. Use tabs to indent code. Ensure the result is printed.",
    "parameters" : {
        "type"       : "object",
        "properties" : {
            "PythonCode" : { 
                "type"       : "string", 
                "description": "The Python code to execute." 
            }
        },
        "required"   : ["PythonCode"]
    }
}, {
    "type"       : "function",
    "name"       : "GetWeather",
    "description": "Retrieve weather information for a specified location.",
    "parameters" : {
        "type"       : "object",
        "properties" : {
            "Location": { 
                "type"       : "string", 
                "description": "The location for which to retrieve weather information."
            }
        },
        "required"   : ["Location"]
    }
}]

print(f"{'=' * 20} Start Completions {'=' * 20}")

PreviousResponseID = None

while True:
    Payload = {
        "model"      : "deepseek",
        "input"      : InputItems,
        "tool_choice": "auto",
    }

    if PreviousResponseID:
        Payload["previous_response_id"] = PreviousResponseID
    else:
        Payload["tools"] = Tools

    Response = requests.post(URL, json = Payload)
    if Response.status_code != 200:
        print("API request failed:", Response.text)
        break

    ResponseBody = Response.json()

    PreviousResponseID = ResponseBody.get("id", None)
    OutputItems        = ResponseBody.get("output", [])

    AssistantTextParts = []
    ReasoningParts     = []
    ToolCalls          = []

    for Item in OutputItems:
        ItemType = Item.get("type", "")
        if ItemType == "message":
            Content = Item.get("content", [])
            if Content:
                for Part in Content:
                    if Part.get("type") == "output_text":
                        AssistantTextParts.append(Part.get("text", ""))
                    else:
                        AssistantTextParts.append(Part.get("refusal", ""))
        elif ItemType == "reasoning":
            Content = Item.get("content", [])
            if Content:
                for Part in Content:
                    if Part.get("type") == "output_text":
                        ReasoningParts.append(Part.get("text", ""))
                    else:
                        ReasoningParts.append(Part.get("refusal", ""))
        
        elif ItemType == "function_call":
            ToolCalls.append(Item)

    AssistantText = "\n".join(AssistantTextParts).strip()
    ReasoningText = "\n".join(ReasoningParts).strip()

    if ToolCalls:
        print(f"\nModel Reasoning: \n{ReasoningText if ReasoningText else 'NO CONTENT'}\n")
        print(f"\nModel Reply: \n{AssistantText if AssistantText else 'NO CONTENT'}\n")
        print(f"Tool Calls: \n{json.dumps(ToolCalls, indent=2)}\n")

        NextInputItems = []
        for ToolCall in ToolCalls:
            FunctionName   = ToolCall["name"]
            Arguments      = ToolCall["arguments"]
            PythonFunction = ToolFunctions.get(FunctionName)

            if not PythonFunction:
                print(f"Unknown tool function: {FunctionName}")
                NextInputItems.append({
                    "type"   : "function_call_output",
                    "call_id": ToolCall["call_id"],
                    "output" : f"Unknown tool function: {FunctionName}"
                })

            else:
                ArgumentsDict = json.loads(Arguments)
                Result        = PythonFunction(ArgumentsDict)
                print(f"Result from {FunctionName}: {Result}")
                NextInputItems.append({
                    "type"   : "function_call_output",
                    "call_id": ToolCall["call_id"],
                    "output" : str(Result)
                })

        InputItems = NextInputItems
        continue

    FinalReasoning = ReasoningText if ReasoningText else "No reasoning generated."
    FinalContent   = AssistantText if AssistantText else "No content generated."
    print(f"\nFinal Model Reasoning: \n{FinalReasoning}\n")
    print(f"\nFinal Model Reply: \n{FinalContent}\n")
    break
