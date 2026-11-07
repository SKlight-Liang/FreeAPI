'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module is used to build tool-use instructions 
and parse tool calls from complete text or incremental response chunks.
It provides a streaming parser that can handle tool calls split across multiple chunks,
as well as a utility to construct tool-call identifiers and generate tool-use prompts.
The streaming parser will maintain its state in different chunks
and finally emit text, tool-name, parameter-delta, and completed-call events.
All the implemented features are designed to be compatible with the native APIs.

However, in practical applications, 
due to the illusion of the model and the mismatch between the system prompt words of Codex and DeepSeek, 
there are errors in tool calls in many cases. This problem will be fixed in the future.

Function Table:
- AutoToolChoice(Tools = None):
Returns the tool selection policy based on the presence of available tools.
- WhethertoParseToolCall(Choice, Tools = None):
Judge whether the tool call should be resolved under the given tool selection strategy.
- BuildToolPrompt(Choice, Tools = None):
Build prompt text for tool call format description and available tool list.
'''

import re
import json

try:
    from json_repair import repair_json
except Exception:
    # Preserve the original fallback when the optional dependency is unavailable.
    def repair_json(Text):
        return Text

# Supported opening and closing markers for tool calls.
TOOL_BEGIN = [ "<tool_call>", "<｜tool▁call▁begin｜>", "<｜｜DSML｜｜tool_call>", "<｜DSML｜tool_calls>"]
TOOL_END   = ["</tool_call>",   "<｜tool▁call▁end｜>", "</tool>", "</tool_use>", "</call>"]

# Define the opening and closing markers for tool call parameters.
PARAMS_BEGIN = "<params>"
PARAMS_DELI  = "{"
PARAMS_END   = "</params>"

# Precompute the maximum lengths of the opening and closing markers,
# which is used to determine whether to keep the possible incomplete tool call tags 
# in the buffer during streaming parsing.
MAX_BEGIN = max(map(len, TOOL_BEGIN))
MAX_CLOSE = max([len(PARAMS_END), *map(len, TOOL_END)])

# Match complete tool calls, including their names and parameter bodies.
# FORMAT: <tool_call>tool_name<params>{"arg":"value"}</params></tool_call>
PAT = re.compile(
    r'(?:' + '|'.join(map(re.escape, TOOL_BEGIN)) + r')\s*([\s\S]*?)\s*'
    r'<params>\s*([\s\S]*?)\s*</params>\s*'
    r'(?:' + '|'.join(map(re.escape, TOOL_END)) + r')'
)

# The calling format description used in the system prompt to instruct the model to call tools correctly.
TOOL_CALL_FORMAT = '''
[important]: When calling a tool, use exactly this format:\n
<tool_call>tool_name<params>{"arg":"value"}</params></tool_call>
'''

# Return the tool selection policy based on the presence of available tools.
def AutoToolChoice(Tools = None):
    if Tools:
        return "auto"
    return "none"

# Determine whether the tool calls should be parsed based on the tool selection policy.
def WhethertoParseToolCall(Choice, Tools = None):
    return bool(Tools) and Choice not in (None, "none")

# Build the prompt describing available tools and any forced tool selection.
def BuildToolPrompt(Choice, Tools = None):
    # If the resolution tool call is not required under the current policy, return an empty string.
    if not WhethertoParseToolCall(Choice, Tools):
        return ""

    # Format the tool list as an itemized description with their names, descriptions, and parameters.
    Descriptions = "\n".join(
        f"- {Tool.get('name')}: {Tool.get('description', '')}\n"
        f"  parameters: {json.dumps(Tool.get('parameters', {}), ensure_ascii = False)}"
        for Tool in (Tools or [])
    )

    # If a specific tool is forced to be called, include that information in the prompt.
    ForcedTool = None
    if isinstance(Choice, dict):
        ForcedTool = Choice.get("name")

    return (
        f"{TOOL_CALL_FORMAT}\nAvailable tools:\n{Descriptions}"
        + (f"\nYou must call: {ForcedTool}" if ForcedTool else "")
    )

'''
ToolCallParser Class

This class is used to parse tool calls from text, 
including handling cases where the tool call is split across multiple chunks.

Functions Table(Public):
- ParseToolCalls(Text):
Resolve all tool calls at once from a complete string.
- BuildCallID(Name, Parameters):
Build a standard format tool call string used for identification or backfill.
- Push(Chunk):
Streaming parse a chunk of text and return a list of events.
- Finish():
Return the final event after all chunks have been processed.
'''

class ToolCallParser:
    # Extract all complete tool calls found in the supplied text.
    @staticmethod
    def ParseToolCalls(Text):
        return [
            {
                "tool":       Match.group(1).strip(),
                "parameters": repair_json(Match.group(2).strip()),
                "start":      Match.start(),
                "raw":        Match.group(0)
            } for Match in PAT.finditer(Text)
        ]

    # Construct a standard format tool call string based on the tool name and parameters.
    # The parameters are truncated to 200 characters if they exceed that length.
    @staticmethod
    def BuildCallID(Name, Parameters):
        # Here we will truncate the parameters to 200 characters if they exceed that length.
        # Because we do not want to generate too long tool call identifiers, 
        # which may cause issues in some contexts.
        ParameterPreview = ""
        if len(Parameters) <= 203:
            ParameterPreview = Parameters
        else:
            ParameterPreview = Parameters[:200] + "..."

        return f"{TOOL_BEGIN[0]}{Name}<params>{ParameterPreview}</params>{TOOL_END[0]}"

    # Initialize the streaming parser with an empty state,
    # including the input buffer, current tool fields, current parameters buffer, and parser state.
    def __init__(self):
        self.Buffer     = ""
        self.Name       = ""
        self.Parameters = ""
        self.State      = "normal"

    # Consume a streaming chunk and parse the event list step by step 
    # according to the current state of the parser.
    def Push(self, Chunk):
        # If the chunk is not empty, append it to the buffer for processing.
        if Chunk:
            self.Buffer += Chunk

        # An list to hold the events generated during parsing of the current buffer.
        Events = []
        while True:
            # NORMAL State: Look for the earliest opening marker for a tool call.
            if self.State == "normal":
                # Find the earliest position of any opening marker in the buffer.
                Matches = []
                for Tag in TOOL_BEGIN:
                    Index = self.Buffer.find(Tag)
                    if Index >= 0:
                        Matches.append((Index, Tag))

                # If no opening marker is found, we need to consider whether to emit any text from the buffer.
                # However, here we do not know if the output has been truncated.
                # It may lead to incomplete tool call tags being split across chunks, 
                # so we need to keep a certain length of the buffer.
                if not Matches:
                    # If the buffer length exceeds the maximum opening marker length, emit the excess text.
                    if len(self.Buffer) >= MAX_BEGIN:
                        EmitLength = len(self.Buffer) - MAX_BEGIN + 1
                        Events.append({
                            "type": "text_delta", 
                            "data": self.Buffer[:EmitLength]
                        })
                        self.Buffer = self.Buffer[EmitLength:]
                    break

                # Select the earliest opening marker and its index in the buffer.
                Index, Tag = min(Matches, key = lambda Item: Item[0])
                # If there is any text before the opening marker, emit it as a text delta event.
                if Index:
                    Events.append({
                        "type": "text_delta", 
                        "data": self.Buffer[:Index]
                    })
                # Remove the opening marker from the buffer and transition to the "name" state.
                self.Buffer = self.Buffer[Index + len(Tag):]
                self.State  = "name"

                continue

            # NAME State: Resolution the tool name from the buffer, 
            # until the opening parameters tag or a bare opening brace is found.
            elif self.State == "name":
                # Find the first occurrence of the opening parameters tag in the buffer.
                Index = self.Buffer.find(PARAMS_BEGIN)
                SeparatorLength = len(PARAMS_BEGIN)
                # If the opening parameters tag is not found, 
                # look for a bare opening brace as an alternative separator.
                if Index < 0:
                    BraceIndex = self.Buffer.find(PARAMS_DELI)
                    if BraceIndex < 0:
                        break

                    # Considering that we need to treat the parameters as a JSON object,
                    # we need to keep the opening brace in the buffer for subsequent parsing.
                    # That is why we set the index to one character before the opening brace.
                    Index = max(0, BraceIndex - 1)
                    SeparatorLength = 1

                # The tool name is just the text before the opening parameters tag or the bare opening brace.
                self.Name = self.Buffer[:Index].strip()
                Events.append({
                    "type": "tool_call_name",
                    "data": self.Name
                })
                # Remove the tool name and the opening parameters tag from the buffer,
                # and transition to the "params" state for parsing the parameters.
                self.Buffer = self.Buffer[Index + SeparatorLength:]
                self.State  = "params"

                continue

            # PARAMS State: Resolution the tool parameters from the buffer,
            # until the closing parameters tag or any closing marker is found.
            elif self.State == "params":
                # Find the first occurrence of the closing parameters tag in the buffer.
                Index = self.Buffer.find(PARAMS_END)
                # Considering that the closing parameters tag may not exist,
                # we need to treat the end markers of tool calls as an alternative closing tag.
                EndMatches = []
                for Tag in TOOL_END:
                    EndIndex = self.Buffer.find(Tag)
                    if EndIndex >= 0:
                        EndMatches.append((EndIndex, Tag))

                # If neither the closing parameters tag nor any closing marker is found,
                # that means the parameters are incomplete and we need to wait for more chunks.
                if Index < 0 and not EndMatches:
                    # We have to avoid the situation where the end marker is truncated across chunks,
                    # so we need to keep a certain length of the buffer.
                    if len(self.Buffer) >= MAX_CLOSE:
                        EmitLength = len(self.Buffer) - MAX_CLOSE + 1
                        Delta = self.Buffer[:EmitLength]

                        Events.append({
                            "type": "tool_call_parameters_delta",
                            "data": Delta
                        })
                        self.Parameters += Delta
                        
                        self.Buffer = self.Buffer[EmitLength:]
                    
                    break

                # If the closing parameters tag is found and it appears before any closing marker,
                # we can consider the parameters to be complete and emit a tool call event.
                if Index >= 0 and (not EndMatches or Index < min(EndMatches)[0]):
                    # Repair the parameters JSON and emit a tool call event with the tool name and parameters.
                    self.Parameters = repair_json(self.Parameters + self.Buffer[:Index]).strip()
                    Events.append({
                        "type": "tool_call",
                        "data": {
                            "name"      : self.Name,
                            "parameters": self.Parameters
                        }
                    })
                    # Reset the tool name and parameters, remove the closing parameters tag from the buffer,
                    # and transition to the "end" state to handle any remaining closing markers.
                    self.Name       = ""
                    self.Parameters = ""
                    self.Buffer     = self.Buffer[Index + len(PARAMS_END):]
                    self.State      = "end"

                    continue

                # Otherwise, we should regard the earliest closing marker as the end of the parameters.
                EndIndex, Tag = min(EndMatches, key = lambda Item: Item[0])
                # Repair the parameters JSON and emit a tool call event with the tool name and parameters.
                self.Parameters = repair_json(self.Parameters + self.Buffer[:EndIndex]).strip()
                Events.append({
                    "type": "tool_call",
                    "data": {
                        "name"      : self.Name,
                        "parameters": self.Parameters
                    }
                })
                # Reset the tool name and parameters, remove the closing marker from the buffer,
                # and transition back to the "normal" state to look for the next tool call.
                self.Name       = ""
                self.Parameters = ""
                self.Buffer     = self.Buffer[EndIndex + len(Tag):]
                self.State      = "normal"

                continue

            # END State: After a completed tool call, we need to discard any closing markers 
            # and transition back to the "normal" state.
            elif self.State == "end":
                # Find the position of the closing marker in the buffer.
                EndMatches = []
                for Tag in TOOL_END:
                    EndIndex = self.Buffer.find(Tag)
                    if EndIndex >= 0:
                        EndMatches.append((EndIndex, Tag))
                # If no closing marker is found, we need to wait for more chunks.
                if not EndMatches:
                    break
                # Otherwise, we can discard the closing marker and transition back to the "normal" state.
                EndIndex, Tag = min(EndMatches, key = lambda Item: Item[0])

                self.Buffer = self.Buffer[EndIndex + len(Tag):]
                self.State  = "normal"

                continue

        return Events

    # Clear the contents of the buffer at the end of a streaming session
    def Finish(self): 
        # If the parser is in the "params" state, 
        # we need to emit a tool call event with the current tool name and parameters.
        if self.State == "params":
            return {
                "type": "tool_call",
                "data": {
                    "name"      : self.Name,
                    "parameters": (self.Parameters + self.Buffer).strip()
                }
            }
        # If there is any remaining text in the buffer, we need to emit a text delta event.
        if self.Buffer:
            return {
                "type": "text_delta",
                "data": self.Buffer
            }
        
        return None
