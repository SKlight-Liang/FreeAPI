'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module is responsible for normalizing OpenAI style requests (completion/response formats) 
into a unified internal request format, and convert the internal response text 
back to the corresponding OpenAI style (message/output).

Functions Table:
- NormalizeCompletions(Messages):
Normalizes OpenAI style chat-completions messages and tool definitions into an internal request dictionary.
- _InputText(Items, Drop = False):
Assembles the input array of responses into a plain text.
- NormalizeResponses(Response):
Normalizes OpenAI style responses format request into an internal request dictionary.
- InternalToCompletions(Text, Match):
Parse the internal response text back to the corresponding completion style assistant message.
- InternalToResponses(Text, Mark, Match):
Parse the internal response text back to the corresponding response style output list.
- UploadAttachmentsCompletions(Request, Client, ModelType):
Upload the images and files embedded in the completions request and return the file ID list.
- UploadAttachmentsResponses(Request, Client, ModelType):
Upload the images and files embedded in the responses request and return the file ID list.
'''

import os
import time
import base64

from .Utils import *
from .Models import *
from .ToolPrompt import *

# Convert chat-completions messages and tool definitions into model input.
def NormalizeCompletions(Messages):
    # Get the model configuration
    Config = GetModelConfig(Messages.get("model", "deepseek"))
    # Extract tool definitions
    Tools  = []
    for Tool in Messages.get("tools", []):
        if Tool.get("type") == "function":
            Tools.append(Tool.get("function", {}))
    # Determine the tool choice
    Choice = Messages.get("tool_choice")
    if not Choice:
        Choice = "auto" if Tools else "none"
    # Build the tool prompt
    ToolPrompt = BuildToolPrompt(Choice, Tools)

    Parts = []
    # Traverse the message list and convert each message into a unified text paragraph.
    # And preserve the original message order and content-part conversion rules as well.
    for Message in Messages.get("messages", []):
        Role    = Message.get("role", "user")
        Content = Message.get("content", "")

        if isinstance(Content, str):
            # If the content is a string, we can directly use it as the text.
            Text = Content
        else:
            # Otherwise, we need to analyze the multimodal content lists.
            Fragments = []
            # Traverse each content part and extract the text, image or file placeholders.
            for Part in Content or []:
                if Part.get("type") == "text":
                    Fragments.append(Part.get("text", ""))
                elif Part.get("type") == "image_url":
                    Fragments.append("<user_image />")
                elif Part.get("type") == "file":
                    Fragments.append(f"<user_file name={Part.get('file', {}).get('filename', 'file')} />")

            # Join the extracted fragments into a single text paragraph.
            Text = "\n".join(Fragments)

        if Role == "tool":
            # If the role is "tool", we need to add a tool call result prefix.
            Parts.append(f"[user tool call result]:\n{Message.get('tool_call_id', '')}\n<output>{Text}</output>")
        else:
            # Otherwise, we can directly add the text paragraph with the role prefix.
            Parts.append(f"[{Role}]:\n{Text}")

    # Join the tool prompt and the message parts into a single message string.
    # The tool prompt is placed at the beginning, followed by the message parts in order.
    CombinedMessage = "\n\n".join(([ToolPrompt] if ToolPrompt else []) + Parts)
    return {**Config, "message": CombinedMessage.strip()}, Choice

# Assemble the input array of responses into a plain text paragraph, 
# while processing message contents, tool results and image/file placeholders.
def _InputText(Items, Drop = False):
    # If the input is a string, we can directly return it.
    if isinstance(Items, str):
        return Items

    Output = []
    # Traverse each input item and convert it into a text paragraph.
    for Item in Items or []:
        # If the item is a tool call output, format it with a special prefix and output tag.
        if Item.get("type") == "function_call_output":
            Output.append(f"[user tool call result]:\n{Item.get('call_id', '')}\n<output>{Item.get('output', '')}</output>")
            continue

        # If the item is not a message, just skip it.
        if Item.get("type") not in (None, "message"):
            continue

        Values  = []
        Content = Item.get("content", "")

        if isinstance(Content, str):
            # If the content is a string, we can directly use it as the value.
            Values = [Content] if Content else []
        else:
            # Otherwise, we need to analyze the multimodal content lists.
            for Part in Content or []:
                if "text" in Part.get("type", ""):
                    Values.append(Part.get("text", ""))
                elif "image" in Part.get("type", ""):
                    Values.append("<user_image />")
                elif "file" in Part.get("type", ""):
                    Values.append(f"<user_file name={Part.get('file_name', 'file')} />")

        # If we have extracted any values, we can add them to the output with the role prefix.
        if Values:
            Output += [f"[{Item.get('role', 'user')}]:", *Values]

    # When drop is True, and the input array has only a single user-role message, 
    # we can remove the sole prefix to avoid redundancy.
    if ( Drop and len(Items or []) == 1 
              and (Items[0].get("type") in (None, "message")) 
              and Items[0].get("role") == "user" and Output):
        Output = Output[1:]

    # Join the output list into a single text paragraph with double newlines.
    return "\n\n".join(Output)

# Convert the responses format request into an internal request dictionary,
# incluing session linkage, tool definitions, and input text assembly.
def NormalizeResponses(Response):
    # Get the model configuration
    Config = GetModelConfig(Response.get("model", "deepseek"))
    # Parse the previous response ID to get the session and message IDs
    PreviousResponseID = Response.get("previous_response_id", "")

    SessionID, MessageID = ParseResponseID("")
    if PreviousResponseID and not PreviousResponseID.startswith("noop-empty-input"):
        SessionID, MessageID = ParseResponseID(PreviousResponseID)

    # Extract tool definitions and transform into internal list
    Tools = []
    for Tool in Response.get("tools", []):
        if Tool.get("type") == "function":
            Tools.append({
                "name": Tool.get("name"),
                "description": Tool.get("description", ""),
                "parameters": Tool.get("parameters", {})
            })
    # Determine the tool choice
    Choice = Response.get("tool_choice")
    if not Choice:
        Choice = "auto" if Tools else "none"
    # Build the tool prompt
    ToolPrompt = BuildToolPrompt(Choice, Tools)

    # Assemble the instructions and input text
    Instructions = ""
    if Response.get("instructions"):
        Instructions = f"[system]:\n{Response['instructions']}"
    Text = _InputText(Response.get("input", []), not ToolPrompt and not Instructions)

    # When tool parsing is enabled and the combined length of instructions and text exceeds 4514 characters,
    # we add a format reminder to the message to ensure proper tool call formatting.
    Emphasis = ""
    if WhethertoParseToolCall(Choice, Tools) and len(Instructions) + len(Text) > 4514:
        Emphasis = TOOL_CALL_FORMAT

    # Combine the tool prompt, instructions, input text, and emphasis into a single message string.
    CombinedMessage = "\n\n".join(Value for Value in (ToolPrompt, Instructions, Text, Emphasis) if Value).strip()
    return {
        **Config, 
        "message"          : CombinedMessage, 
        "session_id"       : SessionID,
        "parent_message_id": MessageID
    }, Choice


# Convert the internal response text back to the corresponding OpenAI style message format,
# containing the role, cleaned content, and tool call information.
def InternalToCompletions(Text, Match):
    # Parse the tool calls from the text if matching is enabled
    Calls = []
    if Match:
        Calls = ToolCallParser.ParseToolCalls(Text)
    # Construct the assistant message infrastructure.
    Message = {
        "role"   : "assistant", 
        "content": Text
    } 

    # If there are tool calls, fill in the tool call information.
    # And replace the tool call spans in the original content with markers for clarity.
    if Calls:
        Message["tool_calls"] = [
            {
                "id"      : ToolCallParser.BuildCallID(Call["tool"], Call["parameters"]),
                "type"    : "function",
                "function": {
                    "name"     : Call["tool"],
                    "arguments": Call["parameters"]
                }
            } for Call in Calls
        ]

        Chunks = []
        Cursor = 0 

        # Traverse the tool calls and replace each accepted call with its marker, 
        # while skipping overlapping spans.
        for Call in Calls:
            # If the start of the current call is before the cursor,
            # it means it overlaps with a previous call, so we skip it.
            if Call["start"] < Cursor:
                continue
            # Extract the text before the current call and add it to the chunks if it's not empty
            Before = Text[Cursor:Call["start"]].strip()
            if Before:
                Chunks.append(Before)
            # Add the tool call marker to the chunks
            Chunks.append(f"<tool_call>{Call['tool']}</tool_call>")
            # Move the cursor to the end of the current call
            Cursor = Call["start"] + len(Call["raw"])

        # Process the remaining text after the last tool call and add it to the chunks if it's not empty
        Tail = Text[Cursor:].strip()
        if Tail:
            Chunks.append(Tail)
        # Join the chunks into a cleaned content string with double newlines
        Cleaned = "\n\n".join(Chunks).strip()
        # If the cleaned content is not empty, update the message content.
        # Otherwise, remove the content key from the message.
        if Cleaned:
            Message["content"] = Cleaned
        else:
            Message.pop("content", None)

    return Message

# Convert the internal response text back to corresponding response style output list.
# That means the messages will be interleaved with the tool call items, 
# and each item will have its own type, id, and status.
def InternalToResponses(Text, Mark, Match):
    # Parse the tool calls from the text if matching is enabled 
    ToolCalls = []
    if Match:
        ToolCalls = ToolCallParser.ParseToolCalls(Text)

    # Transform each tool call into a triple: (Start Position, End Position, Function Call Object)
    Segments = []
    for Call in ToolCalls:
        Segments.append((
            Call["start"],
            Call["start"] + len(Call["raw"]),
            {
                "type"     : "function_call",
                "call_id"  : ToolCallParser.BuildCallID(Call["tool"], Call["parameters"]),
                "name"     : Call["tool"],
                "arguments": Call["parameters"],
                "id"       : "func_call_",
                "status"   : "completed"
            }
        ))

    # Sort the segments by their start position
    Segments.sort()

    Output = []
    Cursor = 0
    # Traverse the segments and interleave the assistant text segments and tool call items
    for Start, End, Item in Segments:
        # If the start of the current segment is before the cursor, 
        # it means it overlaps with a previous segment, so we skip it.
        if Start < Cursor:
            continue
        # Extract the text before the current segment and add it to the output if it's not empty
        TextSegment = Text[Cursor:Start].strip()
        if TextSegment:
            Output.append({
                "type"   : "message",
                "content": [{"type": "output_text", "text": TextSegment, "annotations": []}],
                "role"   : "assistant",
                "id"     : "msg_",
                "status" : "completed"
            })
        # Add the current tool call item to the output
        Output.append(Item)
        # Move the cursor to the end of the current segment
        Cursor = End

    # Process the remaining text after the last segment and add it to the output if it's not empty
    TailSegment = Text[Cursor:].strip()
    if TailSegment:
        Output.append({
            "type"   : "message",
            "content": [{"type": "output_text", "text": TailSegment, "annotations": []}],
            "role"   : "assistant",
            "id"     : "msg_",
            "status" : "completed"
        })

    # Append the mark and index to each item's id for uniqueness
    for Index, Item in enumerate(Output):
        Item["id"] += f"{Mark}_{Index}"

    return Output

# Upload the images and files embedded in the completions request,
# and return the list of uploaded file IDs in order.
async def UploadAttachmentsCompletions(Request, Client, ModelType):
    IDs = []
    # Traverse the messages in the request
    for Message in Request.get("messages", []):
        # If the message content is not a list, skip it
        if not isinstance(Message.get("content"), list):
            continue
        # Traverse the content parts in the message
        for Part in Message["content"]:
            # If the part is an image URL, upload it and append the file ID to the list.
            # Here we will use URL's basename as the filename, or just default to "image.png".
            if Part.get("type") == "image_url":
                URL = Part.get("image_url", {}).get("url")
                if URL:
                    IDs.append(await Client.UploadFile(
                        URL, os.path.basename(URL.split("?", 1)[0]) or "image.png", ModelType
                    ))

            # If the part is a file, upload it and append the file ID to the list.
            elif Part.get("type") == "file":
                File = Part.get("file", {})
                # Firstly, we will check if the file already has a file_id, if so, we just use it.
                if File.get("file_id"):
                    IDs.append(File["file_id"])
                    continue
                # Otherwise, we will check if the file data is provided. 
                # If so, we will decode it and upload it as a file.
                Data = File.get("file_data")
                if Data:
                    EncodedData = ""
                    if isinstance(Data, str) and "," in Data:
                        EncodedData = Data.split(",", 1)[1]
                    else:
                        EncodedData = Data

                    IDs.append(await Client.UploadFile(
                        base64.b64decode(EncodedData),
                        File.get("filename") or f"file_{int(time.time() * 1000)}",
                        ModelType
                    ))
                # If neither file id nor file data is provided, we have to upload file url as a fallback.
                elif File.get("file_url"):
                    IDs.append(await Client.UploadFile(
                        File["file_url"], File.get("filename") or "file", ModelType
                    ))

    return IDs

# Upload the images and files embedded in the responses request,
# and return the corresponding remote file ID list.
async def UploadAttachmentsResponses(Request, Client, ModelType):
    IDs = []
    # We can only process the input items when it is a list, otherwise we just return an empty list.
    if not isinstance(Request.get("input"), list):
        return []

    # Traverse the input items in the request
    # (The following operations are almost the same as the previous ones.)
    for Item in Request.get("input", []):
        # If the item content is not a list, skip it
        if not isinstance(Item.get("content"), list):
            continue
        # Traverse the content parts in the item
        for Part in Item["content"]:
            # If the part already has a file id, we just append it to the list and continue.
            if Part.get("file_id"):
                IDs.append(Part["file_id"])
                continue

            # If the part is an input image, upload it and append the file ID to the list.
            if Part.get("type") == "input_image" and Part.get("image_url"):
                IDs.append(await Client.UploadFile(
                    Part["image_url"],
                    os.path.basename(Part["image_url"].split("?", 1)[0]) or "image.png",
                    ModelType
                ))

            # If the part is an input file, upload it and append the file ID to the list.
            if Part.get("type") == "input_file":
                # If the file data is provided, we will decode and upload it.
                if Part.get("file_data"):
                    IDs.append(await Client.UploadFile(
                        base64.b64decode(Part["file_data"]),
                        Part.get("file_name") or "file", ModelType
                    ))
                # Otherwise, we will check if the file url is provided.
                elif Part.get("file_url"):
                    IDs.append(await Client.UploadFile(
                        Part["file_url"], Part.get("file_name") or "file", ModelType
                    ))

    return IDs
