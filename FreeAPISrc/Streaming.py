'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module is responsible for converting DeepSeek parser events into OpenAI-compatible
event streams. We will provide two asynchronous generators:
- Completions events for /v1/chat/completions style streaming:
    Translate the response into incremental blocks in the form of chat.completion.chunk.
- Response events for /v1/responses style streaming:
    Translate the response into a response API style event stream, including 
    response.created, output.item.added and response.completed events.

Both generators support the ability to parse tool calls from the text stream using the 
ToolCallParser. This enables more advanced interaction between the model and external 
tools, allowing for dynamic tool invocation based on the model's output. Well, at the 
same time, the thinking content of the model will be also included in the output, 
which can be used to provide more context and reasoning behind the model's decisions.
These contents are mapped to reasoning content in the Completions events, or 
reasoning output items in the Response events.

Functions Table:
- _CompletionChunk(RequestID, Model, Delta, Finish = None, Usage = None):
Construct a single chat.completion.chunk incremental block based on the provided parameters.
- CompletionEvents(Stream, RequestID, Model, InputLen, UseTool = False):
Convert raw DeepSeek stream messages into chat.completion.chunk sequences.
- ResponseEvents(Stream, SessionID, Model, InputLen, UseTool = False):
Convert raw DeepSeek stream messages into response API style event streams.
'''

import time

from .Utils import *
from .ToolPrompt import *
from .DeepSeekStreamParser import *

# Construct an OpenAI-style chat.completion.chunk incremental block based 
# on the provided parameters. This is an auxiliary function used internally 
# to format the streaming output in a way that is compatible with OpenAI's API.
def _CompletionChunk(RequestID, Model, Delta, Finish = None, Usage = None):
    return {
        "id"     : RequestID,
        "object" : "chat.completion.chunk",
        "created": int(time.time()),
        "model"  : Model,
        "choices": [{
            "index"        : 0,
            "delta"        : Delta,
            "finish_reason": Finish,
            "logprobs"     : None,
        }],
        "usage"  : Usage,
    }

# Convert raw DeepSeek stream messages into chat.completion.chunk sequences, including 
# reasoning content and tool calls. This function is designed to be used as an asynchronous 
# generator, yielding each chunk as it is processed. We will preserve the original callback 
# queue, parser finalization, and output order to ensure that the resulting stream is 
# consistent with the input.
async def CompletionEvents(Stream, RequestID, Model, InputLen, UseTool = False):
    # Queue: Store parser deltas until they can be processed and emitted in order.
    Queue      = []
    # ToolParser: If tool parsing is enabled, initialize a ToolCallParser to handle tool calls.
    ToolParser = ToolCallParser() if UseTool else None
    # ToolCount: Keep track of the number of tool calls processed to assign unique indices.
    ToolCount  = 0

    # INCREMENTAL CALLBACK: Handle each parser delta as it is produced, buffering them in the 
    # Queue for later processing. This allows us to maintain the order of events and ensure 
    # that they are emitted in the correct sequence.
    def DeltaHandler(EventType, Data):
        # Append the event type and data to the Queue
        Queue.append((EventType, Data))

    # Create a DeepSeekStreamParser instance registered with the DeltaHandler 
    # callback to process incoming stream data.
    Parser = DeepSeekStreamParser(DeltaHandler)
    # Generate an initial chunk indicating the start of the chat completion stream, 
    # with the role set to "assistant". This is a standard practice in OpenAI's chat
    # completion API to signal the beginning of the assistant's response.
    yield _CompletionChunk(RequestID, Model, {"role": "assistant"})

    # DELTA CONVERSION: Convert each parser delta into one or more chat.completion.chunk blocks,
    # including reasoning content and tool calls. This function is responsible for translating
    # the internal event representation into the format expected by OpenAI's API, ensuring that
    # all relevant information is preserved and correctly structured.
    def DeltaConversion(EventType, Data):
        nonlocal ToolCount

        # For thinking content, we will map it to reasoning content increments. 
        if EventType == "THINK":
            return [_CompletionChunk(RequestID, Model, {"reasoning_content": Data})]
        # If tool parsing is not enabled, we will regard all other events as text deltas and emit 
        # them as chat.completion.chunk blocks with the content field set to the event data.
        if not ToolParser:
            return [_CompletionChunk(RequestID, Model, {"content": Data})]
    
        # Otherwise, we need to use the ToolCallParser to parse the event data and extract any tool calls.
        Chunks = []
        # Push the event data into the ToolCallParser, which will return a sequence of 
        # events representing the parsed tool calls and any remaining text deltas. We 
        # will iterate over these events and convert them into chat.completion.chunk blocks.
        for Event in ToolParser.Push(Data):
            # Depending on the event type, we will either emit a text delta or a tool call.
            if Event["type"] == "text_delta":
                Chunks.append(_CompletionChunk(RequestID, Model, {"content": Event["data"]}))
            elif Event["type"] == "tool_call":
                # If the event is a tool call, we need to build a tool call block with the data provided by 
                # the ToolCallParser. This includes the tool name, parameters, and a unique index for the call.
                ToolData = Event["data"]
                # Construct a tool call block with the appropriate index, ID, type, and function details.
                ToolCall = {
                    "index"   : ToolCount,
                    "id"      : ToolCallParser.BuildCallID(ToolData["name"], ToolData["parameters"]),
                    "type"    : "function",
                    "function": {
                        "name"     : ToolData["name"],
                        "arguments": ToolData["parameters"],
                    },
                }
                # Append the tool call block to the list of chunks to be emitted
                Chunks.append(_CompletionChunk(RequestID, Model, {"tool_calls": [ToolCall]}))
                # Increment the ToolCount
                ToolCount += 1

        return Chunks
        
    # Traverse the original blocks in the response stream, converting each raw fragment 
    # into parser events and emitting the resulting chat.completion.chunk blocks.
    async for Raw in Stream:
        # Decode each raw fragment and push it into the DeepSeekStreamParser, 
        # which will invoke the DeltaHandler
        Parser.Push(Raw.decode(errors = "replace") if isinstance(Raw, bytes) else Raw)  
        # Emit any queued deltas that have been produced by the parser in time, converting 
        # them into chat.completion.chunk blocks using the DeltaConversion function.
        while Queue:
            # Pop the next event from the Queue
            EventType, Data = Queue.pop(0)
            # Convert it into one or more chat.completion.chunk blocks
            for Chunk in DeltaConversion(EventType, Data):
                # Yield each chunk to the caller of the CompletionEvents generator
                yield Chunk

    # After the stream has been fully processed, we need to finalize the parser and emit 
    # any remaining events. This includes flushing any pending deltas and handling any 
    # final tool calls that may have been produced by the ToolCallParser.             
    Parser.Finish()
    # Emit any remaining queued deltas after the parser has been finalized
    while Queue:
        # Pop the next event from the Queue
        EventType, Data = Queue.pop(0)
        # Convert it into one or more chat.completion.chunk blocks
        for Chunk in DeltaConversion(EventType, Data):
            # Yield each chunk to the caller of the CompletionEvents generator
            yield Chunk

    # Similar operations should also be performed on the ToolCallParser if it is enabled, 
    # ensuring that any final tool calls are emitted before the completion of the stream. 
    if ToolParser:
        Event = ToolParser.Finish()

        # If the ToolCallParser has produced any final events, we need to convert them into
        # chat.completion.chunk blocks and yield them to the caller.
        if Event:
            # Depending on the event type, we will either emit a text delta or a tool call.
            if Event["type"] == "text_delta":
                yield _CompletionChunk(RequestID, Model, {"content": Event["data"]})
            elif Event["type"] == "tool_call":
                # If the event is a tool call, we need to build a tool call block with the data provided by 
                # the ToolCallParser. This includes the tool name, parameters, and a unique index for the call.
                ToolData = Event["data"]
                # Construct a tool call block with the appropriate index, ID, type, and function details.
                ToolCall = {
                    "index"   : ToolCount,
                    "id"      : ToolCallParser.BuildCallID(ToolData["name"], ToolData["parameters"]),
                    "type"    : "function",
                    "function": {
                        "name"     : ToolData["name"],
                        "arguments": ToolData["parameters"],
                    },
                }
                # Yield the final tool call block to the caller of the CompletionEvents generator
                yield _CompletionChunk(RequestID, Model, {"tool_calls": [ToolCall]})
                # Increment the ToolCount
                ToolCount += 1

    # After all events have been processed and emitted, we will estimate the total token usage.
    # The specific calculation method could be found in the .Utils module.
    Response   = Parser.Text("response").strip()
    Thinking   = Parser.Text("think").strip() or Parser.Text("thinking").strip()
    TokenUsage = EstimateTokenUsage(Parser.TokenUsage, InputLen, len(Response) + len(Thinking))

    # Yield a final chat.completion.chunk block indicating the end of the stream,
    # including the total token usage and the breakdown of input and output tokens.
    yield _CompletionChunk(
        RequestID, Model, {}, 
        "tool_calls" if ToolCount else "stop", {
            "total_tokens"     : TokenUsage["total_tokens"],
            "prompt_tokens"    : TokenUsage["input_tokens"],
            "completion_tokens": TokenUsage["output_tokens"],
        }
    )

# Convert raw DeepSeek stream messages into a response API style event sequence. Same as above,
# we can also handle tool calls and reasoning output items, but the output format is different.
# Additionally, the response flow could maintain a persistent conversation state, allowing for 
# more complex interactions between the model and external tools. This means that we need to 
# preserve more information about the response lifecycle, including the original final IDs, 
# output item transitions and so on.
async def ResponseEvents(Stream, SessionID, Model, InputLen, UseTool = False):
    # Queue: Store parser deltas until they can be processed and emitted in order.
    Queue      = []
    # Ready: A mutable container to hold the response message ID once the ready event is received.
    Ready      = [None]
    # ToolParser: If tool parsing is enabled, initialize a ToolCallParser to handle tool calls.
    ToolParser = ToolCallParser() if UseTool else None
    # SequenceID: Keep track of the sequence number for each event emitted, 
    # ensuring that they are ordered correctly.
    SequenceID = 1
    # Created: A flag to indicate whether the response.created event has been emitted.
    Created    = False
    # Outputs: A list to hold the output items that will be included in the final response.
    Outputs    = []
    # Current: A variable to track the current output item type, allowing us to create new items
    # whenever the type changes (e.g., from message to reasoning).
    Current    = None

    # INCREMENTAL CALLBACK: Handle each parser delta as it is produced, buffering them in the 
    # Queue for later processing. This allows us to maintain the order of events and ensure 
    # that they are emitted in the correct sequence.
    def DeltaHandler(EventType, Data):
        Queue.append((EventType, Data))

    # READY CALLBACK: Handle the ready event, which provides the response message ID.
    # This ID will be used to construct the final response ID, which is a combination 
    # of the session ID and the message ID. The ready event is important because it 
    # signals that the model is ready to start producing output.
    def ReadyHandler(Event):
        if Event.get("event") == "ready":
            Ready[0] = (Event.get("data") or {}).get("response_message_id")
        
    # Create a DeepSeekStreamParser instance registered with the DeltaHandler and ReadyHandler
    # callbacks to process incoming stream data. The ReadyHandler will be invoked whenever
    # a ready event is received, allowing us to capture the response message ID for later use.
    Parser = DeepSeekStreamParser(DeltaHandler, ReadyHandler)

    # RESPONSEID CONSTRUCTION: Construct the response ID based on the session ID and the message ID
    # received from the ready event. If the message ID is not available, we will use only the session 
    # ID to maintain a consistent identifier for the response throughout its lifecycle.
    def CurrentResponseID():
        return ConstructResponseID(
            SessionID, 
            Ready[0] if Ready[0] is not None else Parser.MessageID
        )
    
    # TEXT OUTPUT: Create a new output item whenever its kind changes. This function is responsible for
    # managing the output items in the response, ensuring that each item is correctly structured and
    # that the appropriate events are emitted to signal the addition and completion of each item. We 
    # will distinguish message and reasoning content according to the Kind parameter, and generate
    # the corresponding events for each output item, including added and delta events.
    def AddText(Kind, Data):
        nonlocal Current
        nonlocal SequenceID

        Events = []

        # If the current output item type is different from the new kind, we need to create 
        # a new output item and emit the corresponding events to signal its addition.
        if Current != Kind:
            # Update the current output item type to the new kind
            Current = Kind
            # Determine the index of the new output item based on the current length of the Outputs list
            Index = len(Outputs)
            # Create a new output item
            Item = {
                "type"   : Kind,
                "content": [],
                "role"   : "assistant",
                "id"     : f"msg_{CurrentResponseID()}_{Index}",
                "status" : "in_progress",
            }
            # Append the new output item to the Outputs list and emit the corresponding events
            Outputs.append(Item)
            Events.append({
                "type"           : "response.output_item.added",
                "output_index"   : Index,
                "item"           : Item.copy(),
                "sequence_number": SequenceID
            })
            # Increment the sequence number for the next event
            SequenceID += 1
    
            # Create a new content part for the output item
            Part = {
                "type"        : "output_text",
                "text"        : "",
                "annotations" : [],
            }
            # Append the new content part to the output item and emit the corresponding event
            Item["content"].append(Part)
            Events.append({
                "type"           : "response.content_part.added",
                "item_id"        : Item["id"],
                "output_index"   : Index,
                "content_index"  : 0,
                "part"           : Part.copy(),
                "sequence_number": SequenceID,
            })
            # Increment the sequence number for the next event
            SequenceID += 1

        # Add the text delta to the last content part of the current output item
        Item = Outputs[-1]
        Item["content"][0]["text"] += Data
        # Emit the corresponding event for the text delta
        Events.append({
            "type"           : "response.output_text.delta",
            "item_id"        : Item["id"],
            "output_index"   : len(Outputs) - 1,
            "content_index"  : 0,
            "delta"          : Data,
            "sequence_number": SequenceID,
        })
        # Increment the sequence number for the next event
        SequenceID += 1

        return Events        

    # TOOL OUTPUT: Create a new output item for a tool call, including the function name and parameters.
    # This function will generate the necessary events to signal the addition of the tool call output item, 
    # as well as the completion of the function call arguments and the finalization of the output item.
    def AddTool(ToolData):
        nonlocal Current
        nonlocal SequenceID
        
        # Set the current output item type to "FUNC" to indicate that we are processing a tool call.
        Current = "FUNC"
        # Determine the index of the new output item based on the current length of the Outputs list.
        Index   = len(Outputs)
        # Create a new output item for the tool call, including the function name and parameters.
        Item    = {
            "type"     : "function_call",
            "name"     : ToolData["name"],
            "arguments": ToolData["parameters"],
            "id"       : f"func_call_{CurrentResponseID()}_{Index}",
            "call_id"  : ToolCallParser.BuildCallID(ToolData["name"], ToolData["parameters"]),
            "status"   : "completed",
        }
        # Append the new output item to the Outputs list
        Outputs.append(Item)
        
        # Generate the events to signal the addition of the tool call output item.
        AddedItem = {
            **Item, 
            "status"   : "in_progress", 
            "arguments": ""
        }
        Events    = [{
            "type"           : "response.output_item.added",
            "output_index"   : Index,
            "item"           : AddedItem,
            "sequence_number": SequenceID
        }]
        # Increment the sequence number for the next event
        SequenceID += 1
        
        # Generate the event to signal the completion of the function call arguments,
        # indicating that the tool call has been fully specified and is ready for execution.
        Events.append({
            "type"           : "response.function_call_arguments.done",
            "output_index"   : Index,
            "item_id"        : Item["id"],
            "name"           : Item["name"],
            "arguments"      : Item["arguments"],
            "sequence_number": SequenceID,
        })
        # Increment the sequence number for the next event
        SequenceID += 1
                
        # Generate the event to signal the finalization of the output item, 
        # indicating that the tool call has been completed.
        Events.append({
            "type"           : "response.output_item.done",
            "output_index"   : Index,
            "item"           : Item,
            "sequence_number": SequenceID,
        })
        # Increment the sequence number for the next event
        SequenceID += 1

        return Events

    # DELTA CONVERSION: Convert each single increment into an OpenAI-style response event list,
    # including reasoning content and tool calls. Among them, reasoning content will be mapped 
    # to reasoning output items, and tool calls will be processed by the ToolCallParser to 
    # extract the function name and parameters.
    def DeltaConversion(EventType, Data):
        # For thinking content, just map it to reasoning text.
        if EventType == "THINK":
            return AddText("reasoning", Data)
        # If tool parsing is not enabled, regard all other events as text deltas 
        # and emit them as message text.
        if not ToolParser:
            return AddText("message", Data)

        # Otherwise, we need to use the ToolCallParser to parse the event data and extract any tool calls.
        Events = []
        # Push the event data into the ToolCallParser, which will return a sequence of 
        # events representing the parsed tool calls and any remaining text deltas. We
        # will iterate over these events and convert them into response events.
        for ParsedEvent in ToolParser.Push(Data):
            # Depending on the event type, we will either emit a text delta or a tool call
            if ParsedEvent["type"] == "text_delta":
                Events += AddText("message", ParsedEvent["data"])
            elif ParsedEvent["type"] == "tool_call":
                Events += AddTool(ParsedEvent["data"])

        return Events

    # Iterate over the raw stream
    async for Raw in Stream:
        # Push each fragment into the DeepSeekStreamParser after decoding it, 
        # which will invoke the DeltaHandler and ReadyHandler callbacks.
        Parser.Push(Raw.decode(errors = "replace") if isinstance(Raw, bytes) else Raw)
        # Once the ready event has been received and the create response event has not 
        # yet been emitted, we will emit it to signal the start of the response stream.
        if not Created and Ready[0] is not None:
            # Update the Created flag to indicate that the response.created event has been emitted
            Created = True
            # Emit the response.created event with the current response ID, model, and initial status.
            yield {
                "type"           : "response.created",
                "response"       : {
                    "id"         : CurrentResponseID(),
                    "object"     : "response",
                    "created_at" : int(time.time()),
                    "status"     : "in_progress",
                    "model"      : Model,
                    "output"     : [],
                    "usage"      : None,
                },
                "sequence_number": SequenceID,
            }
            # Increment the sequence number for the next event
            SequenceID += 1
    
        # Emit any queued deltas that have been produced by the parser in time, converting
        # them into response events using the DeltaConversion function.
        while Queue:
            # Pop the next event from the Queue
            EventType, Data = Queue.pop(0)
            # Convert it into one or more response events
            for Event in DeltaConversion(EventType, Data):
                # Yield each event to the caller of the ResponseEvents generator
                yield Event
            
    # After the stream has been fully processed, we need to finalize the parser and emit
    # any remaining events. This includes flushing any pending deltas and handling any
    # final tool calls that may have been produced by the ToolCallParser.
    Parser.Finish()
    while Queue:
        # Pop the next event from the Queue
        EventType, Data = Queue.pop(0)
        # Convert it into one or more response events
        for Event in DeltaConversion(EventType, Data):
            # Yield each event to the caller of the ResponseEvents generator
            yield Event
    
    # Finalize the ToolCallParser if it is enabled, emitting any remaining tool call events.
    if ToolParser:
        ParsedEvent = ToolParser.Finish()
        # If the ToolCallParser has produced any final events, we need to convert them into
        # response events and yield them to the caller.
        if ParsedEvent:
            # Depending on the event type, we will either emit a text delta or a tool call.
            if ParsedEvent["type"] == "text_delta":
                Events = AddText("message", ParsedEvent["data"])
            else:
                Events = AddTool(ParsedEvent["data"])

            for Event in Events:
                yield Event

    # Construct the final response ID based on the session ID and 
    # the message ID received from the ready event.
    ResponseID = ConstructResponseID(SessionID, Parser.MessageID)

    # Although I do not know why this situation will appear, but if the response.created 
    # event has not been generated before, we will resend it here to ensure that the response 
    # lifecycle is complete. This is a safeguard to ensure that the response.created 
    # event is always emitted, even if the ready event was not received in time. 
    if not Created:
        yield {
            "type"           : "response.created",
            "response"       : {
                "id"         : ResponseID,
                "object"     : "response",
                "created_at" : int(time.time()),
                "status"     : "in_progress",
                "model"      : Model,
                "output"     : [],
                "usage"      : None,
            },
            "sequence_number": SequenceID,
        }
        # Increment the sequence number for the next event
        SequenceID += 1

    # Finalize the output items by marking them as completed and emitting the corresponding events.
    for Index, Item in enumerate(Outputs):
        # Only message and reasoning output items are considered for completion, as they represent
        # the final text output of the model. Other types of output items, such as tool calls, 
        # are not included in this finalization step, as they have already been marked as completed 
        # when they were added.
        if Item["type"] not in ("message", "reasoning"):
            continue
                
        # Mark the output item as completed and emit the corresponding events for each content part.
        Item["status"] = "completed"

        # For each content part in the output item, we will emit a response.output_text.done event
        # followed by a response.content_part.done event. This ensures that each part of the output
        # is correctly finalized and that the sequence of events is preserved.
        for ContentIndex, Part in enumerate(Item["content"]):
            yield {
                "type"           : "response.output_text.done",
                "output_index"   : Index,
                "item_id"        : Item["id"],
                "text"           : Part["text"],
                "sequence_number": SequenceID,
            }
            SequenceID += 1

            yield {
                "type"           : "response.content_part.done",
                "item_id"        : Item["id"],
                "output_index"   : Index,
                "content_index"  : ContentIndex,
                "part"           : Part,
                "sequence_number": SequenceID,
            }
            SequenceID += 1

        # After all content parts have been finalized, we will emit a response.output_item.done event
        # to signal that the output item has been fully completed. 
        yield {
            "type"           : "response.output_item.done",
            "output_index"   : Index,
            "item"           : Item,
            "sequence_number": SequenceID,
        }
        SequenceID += 1
            
    # After all output items have been finalized, we will estimate the total token usage 
    # for the response, including the input tokens and the output tokens.   
    Response   = Parser.Text("response").strip()
    Thinking   = Parser.Text("think").strip() or Parser.Text("thinking").strip()
    TokenUsage = EstimateTokenUsage(Parser.TokenUsage, InputLen, len(Response) + len(Thinking))

    # Construct the final response.completed event, including the total token usage and the
    # breakdown of input and output tokens. This event signals the end of the response stream.
    Result     = {
        "id"         : ResponseID,
        "object"     : "response",
        "created_at" : int(time.time()),
        "status"     : "completed",
        "model"      : Model,
        "output"     : Outputs,
        "usage"      : TokenUsage,
    }

    yield {
        "type"           : "response.completed",
        "response"       : Result,
        "sequence_number": SequenceID,
    }
