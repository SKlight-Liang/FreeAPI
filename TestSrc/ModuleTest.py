import json
import base64
import struct
import asyncio
import aiohttp
import importlib

from pytest import MonkeyPatch

from FreeAPISrc.Utils import *
from FreeAPISrc.Normalize import *
from FreeAPISrc.ToolPrompt import *
from FreeAPISrc.DeepSeekStreamParser import *
from FreeAPISrc.DeepSeekWebClient import *

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

# Tests for the DeepSeekWebClient class.
def WASMTIMEEnginePipelineTest(monkeypatch):
    Calls = {}

    class Engine: pass
    class Store: 
        def __init__(self, Engine): 
            Calls['store_engine'] = Engine
    class Memory:
        def __init__(self): 
            self.Buffer = bytearray(4096)
        def write(self, Store, Data, Pointer):
            self.Buffer[Pointer : Pointer + len(Data)] = Data
        def read(self, Store, Start, End):
            return bytes(self.Buffer[Start : End])

    TestMemory = Memory()
    
    class Func:
        def __init__(self, Kind): 
            self.Kind = Kind
        def __call__(self, Store, *Args):
            if self.Kind == 'alloc': 
                return 128
            if self.Kind == 'stack': 
                return 256 if Args[0] == -16 else 0
            if self.Kind == 'solve':
                Ret                                   = Args[0]
                TestMemory.Buffer[Ret : Ret + 4]      = struct.pack('<i', 1)
                TestMemory.Buffer[Ret + 8 : Ret + 16] = struct.pack('<d', 42.0)
    class Module:
        @classmethod
        def from_file(cls, Engine, Path): 
            Calls['module_engine'] = Engine
            Calls['path']          = Path
            return cls()
    class Exports(dict):
        def __iter__(self): 
            return super().__iter__()
    class Instance:
        def __init__(self, Store, Module, Imports): pass
        def exports(self, Store):
            return Exports(
                memory                          = TestMemory, 
                __wbindgen_export_0             = Func('alloc'), 
                __wbindgen_add_to_stack_pointer = Func('stack'), 
                wasm_solve                      = Func('solve')
            )

    ClientModule = importlib.import_module(DeepSeekWebClient.__module__)
    monkeypatch.setattr(ClientModule, "Engine"  , Engine)
    monkeypatch.setattr(ClientModule, "Store"   , Store)
    monkeypatch.setattr(ClientModule, "Module"  , Module)
    monkeypatch.setattr(ClientModule, "Instance", Instance)

    Payload = {
        "challenge" : "abc",
        "salt"      : "s",
        "expire_at" : 1,
        "difficulty": 2
    }

    Client = object.__new__(DeepSeekWebClient)
    Answer = Client._PoWChallengeSolver(Payload)

    assert Answer == 42
    assert Calls['store_engine'] is Calls['module_engine']

def PoWResponseBuildTest():
    Payload = {
        "algorithm"  : "DeepSeekHashV1",
        "challenge"  : "abc",
        "difficulty" : 144000,
        "salt"       : "salt",
        "signature"  : "sig",
        "expire_at"  : 1234567890
    }

    Client  = object.__new__(DeepSeekWebClient)
    Encoded = Client._BuildPowResponse(Payload, "/api/v0/chat/completion", 42.0)
    Raw     = base64.b64decode(Encoded).decode("utf-8")
    Parsed  = json.loads(Raw)

    assert Parsed == {**Payload, "answer": 42, "target_path": "/api/v0/chat/completion"}
    assert '"answer":42' in Raw and '"answer":42.0' not in Raw

def ChatCompletionsTest():
    async def _Run():
        Client = DeepSeekWebClient({
            "cookie"    : "x",
            "userAgent" : "ua"
        })

        try:
            assert "Content-Type" not in Client.Http.headers

            Form = aiohttp.FormData()
            Form.add_field("file", b"hello", filename = "a.txt", content_type = "text/plain")
            Payload = Form()

            ContentType = str(Payload.headers["Content-Type"])

            assert ContentType.startswith("multipart/form-data;")
            assert "boundary=" in ContentType

        finally:
            await Client.Close()

    asyncio.run(_Run())

if __name__ == "__main__":
    print("Running tests for Utils...")
    ResponseIDTest()
    print("Running tests for DeepSeekStreamParser...")
    SSEParserTest()
    print("Running tests for DeepSeekWebClient...")
    with MonkeyPatch.context() as monkeypatch:
        WASMTIMEEnginePipelineTest(monkeypatch)
    PoWResponseBuildTest()
    ChatCompletionsTest()
    print("Running tests for ToolPrompt...")
    ToolParserTest()
    StreamingToolParserTest()
    print("Running tests for Normalize...")
    CompletionsNormalizeTest()
    ResponseNormalizeTest()
    print("All tests passed!")
