from FreeAPISrc.Utils import *
from FreeAPISrc.Normalize import *
from FreeAPISrc.ToolPrompt import *
from FreeAPISrc.DeepSeekStreamParser import *

# Tests for utility functions in the Utils module.
def ResponseIDTest():
    assert ParseResponseID("1234567890|42")      == ("1234567890", 42)
    assert ConstructResponseID("1234567890", 42) == "1234567890|42"

# Tests for the DeepSeekStreamParser class.
def SSEParserTest():
    Parser      = DeepSeekStreamParser()
    TestPayload = (
        'event: message\n'
        'data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"hi"}]}}}\n'
        '\n'
    )

    Parser.Push(TestPayload)
    assert Parser.Text() == "hi"

# Tests for the ToolCallParser class.
def ToolParserTest():
    TestPayload = "<tool_call>calc<params>{\"x\":1}</params></tool_call>"

    ToolCalls = ToolCallParser.ParseToolCalls(TestPayload)
    assert ToolCalls[0]["tool"] == "calc"

def StreamingToolParserTest():
    Parser  = ToolCallParser()
    Events  = []
    Payload = [
        "hello <tool_",
        "call>exec_js<par",
        "ams>{\"javascript_",
        "code\":\"console.log(89)\"}</params></tool_",
        "call> tail"
    ]
    
    for Chunk in Payload:
        Events += Parser.Push(Chunk)
    Event = Parser.Finish()
    if Event:
        Events.append(Event)

    Calls = [Event for Event in Events if Event["type"] == "tool_call"]
    Text  = "".join(Event["data"] for Event in Events if Event["type"] == "text_delta")

    assert len(Calls) == 1 
    assert Calls[0]["data"]["name"] == "exec_js"
    assert "console.log(89)" in Calls[0]["data"]["parameters"]
    assert Text == "hello  tail"

# Tests for utility functions in the Normalize module.
def CompletionsNormalizeTest():
    Payload = "before <tool_call>calc<params>{'x':1}</params></tool_call> after"
    
    Message = InternalToCompletions(Payload, True)

    assert Message["tool_calls"][0]["function"]["name"] == "calc"
    assert Message["content"] == "before\n\n<tool_call>calc</tool_call>\n\nafter"

def ResponseNormalizeTest():
    Payload = {
        "model": "deepseek",
        "previous_response_id": "noop-empty-input",
        "input": "hi"
    }

    Normalized, _ = NormalizeResponses(Payload)

    assert Normalized["session_id"] is None
    assert Normalized["parent_message_id"] is None


if __name__ == "__main__":
    print("Running tests for Utils...")
    ResponseIDTest()
    print("Running tests for DeepSeekStreamParser...")
    SSEParserTest()
    print("Running tests for ToolPrompt...")
    ToolParserTest()
    StreamingToolParserTest()
    print("Running tests for Normalize...")
    CompletionsNormalizeTest()
    ResponseNormalizeTest()
    print("All tests passed!")
