'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module implements a simple HTTP/WebSocket server serving as a compatibility 
layer for the DeepSeek backend. It is designed to be used with the FreeAPI framework, 
providing a bridge between the DeepSeek web-based client and OpenAI compatible 
interfaces, allowing users to interact with the DeepSeek backend through standard 
OpenAI API calls, including:
/v1/models
/v1/chat/completions
/v1/responses (including SSE streaming)
/v1/responses WebSocket Channel (including SSE streaming)

The server will validate peer access, normalize chat and Responses API requests, 
forward work to the configured backend client, and stream or package results. 
It is designed to be lightweight and easy to deploy, making it suitable for local 
development and testing of OpenAI-compatible applications. Additionally, we also
provide other useful features, such as IP whitelist, CORS middleware, error handling,
queue serialization and session lifecycle management.
'''

import os
import json
import time
import asyncio
import argparse
import ipaddress

from aiohttp import web
from aiohttp import WSMsgType

from .Utils import *
from .Models import *
from .Streaming import *
from .Normalize import *
from .ToolPrompt import *
from .DeepSeekAuth import *
from .ServerClient import *
from .DeepSeekStreamParser import *

# Check whether the request is allowed based on the source IP address and the configured IP allowlist. 
# The server will only accept requests from loopback addresses (127:0.0.1, ::1) and any additional IP 
# addresses specified in the FREEAPI_IP_ALLOWLIST environment variable.
def IPAllowed(Request):
    # Get the remote IP address of the request and normalize it to a standard format.
    RemoteIP = (Request.remote or "").strip()
    # Normalize IPv6-mapped IPv4 addresses and remove any zone identifiers.
    if RemoteIP.startswith("::ffff:"):
        RemoteIP = RemoteIP[7:]
    # Remove any zone identifiers from IPv6 addresses (e.g., %eth0).
    if '%' in RemoteIP:
        RemoteIP = RemoteIP.split('%', 1)[0]
    # Get the set of allowed IP addresses from the environment variable.
    WhiteList = {"127.0.0.1", "::1"} | {
        AllowedIP.strip() for AllowedIP in os.getenv("FREEAPI_IP_ALLOWLIST", "").split(",") 
            if AllowedIP.strip()
    }
    # Check if the remote IP address is in the allowlist.
    return RemoteIP in WhiteList
    
# CORS MIDDLEWARE: Call the downstream request handler to process the request 
# and then add default cross-domain headers to the response. This allows the server 
# to accept requests from any origin, including those from web browsers running on 
# different domains, which is useful for web-based clients that need to interact 
# with the server from a different origin.
@web.middleware
async def CORSMiddleware(Request, handler):
    # Call the downstream request handler to process the request and get the response.
    Response = await handler(Request)
    # Add default cross-domain headers to the response to allow requests from any origin.
    Response.headers.setdefault("Access-Control-Allow-Origin" , "*")
    Response.headers.setdefault("Access-Control-Allow-Headers", "content-type, authorization, x-requested-with")
    Response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    # Return the modified response with the added headers.
    return Response

# IP WHITELIST MIDDLEWARE: Reject disallowed remote peers before executing the downstream handler.
# If the request's source IP address is not in the allowlist, the server will return a 403 Forbidden 
# response. This middleware is useful for restricting access to the server to only trusted clients,
# such as those running on the same machine or within a specific network.
@web.middleware
async def IPWhiteListMiddleware(Request, handler):
    # Check whether the request is allowed
    if not IPAllowed(Request):
        # If the request is not allowed, return a 403 Forbidden response with an error message.
        return web.json_response({
            "error": { "message": "Forbidden" }
        }, status = 403)
    # Otherwise, hand it over to the next processor in the middleware chain or the final request handler.
    return await handler(Request)

# ERROE HANDLER: Wrap the captured exceptions from the downstream handler as a JSON response 
# with an appropriate HTTP status code. Among the exceptions, ValueError will be treated as 
# a 400 Bad Request, while other exceptions will be treated as a 500 Internal Server Error.
async def ErrorHandler(Request, Handler):
    try:
        # Call the downstream request handler to process the request and get the response.
        return await Handler(Request)
    except ValueError as Except:
        # If a ValueError is raised, return a 400 Bad Request response with an error message.
        print(f"FREEAPI > ValueError: {Except} Request: {Request.method} {Request.path_qs}")
        return web.json_response({
            "error": { "message": str(Except), "type": "invalid_request_error" }
        }, status = 400)
    
    except Exception as Except:
        # If any other exception is raised, return a 500 Internal Server Error response with an error message.
        print(f"FREEAPI > Exception: {Except} Request: {Request.method} {Request.path_qs}")
        return web.json_response({
            "error": { "message": str(Except), "type": "server_error" }
        }, status = 500)

# The following constant is used as a placeholder representing empty input in the communication 
# between the server and the browser. It is used to indicate that there is no meaningful input
# data being sent, allowing the server to handle such cases appropriately without confusion or 
# misinterpretation, especially in WebSocket scenarios when there is no actual input to process.
READY_RESPONSE_ID = "noop-empty-input"

'''
ResponseSessionQueue Class
This class is designed to manage and serialize operations for different sessions, 
ensuring that each session's requests are processed in a controlled manner. It 
maintains a queue for each session, enforcing a maximum size limit to prevent overload.
The class provides methods to run operations within the session's context and to clear 
the session state when no longer needed. 

By using asyncio locks, it ensures that operations for a session are executed sequentially, 
preventing race conditions and maintaining the integrity of session data. This is particularly 
useful in scenarios where multiple requests may be made to the same session concurrently, 
such as in a web server handling multiple client connections.
'''
class ResponseSessionQueue:
    # Initialize the ResponseSessionQueue with a maximum queue size for each session.
    def __init__(self, MaxSize=8):
        # Store the maximum queue size
        self.MaxSize = MaxSize
        # Initialize a dictionary to hold the state of each session, 
        # including its lock, waiting count, and last operation.
        self.States  = {}
        # Initialize an asyncio lock to guard access to the session states, 
        # ensuring thread-safe operations.
        self.Guard   = asyncio.Lock()

    # Run an operation for a specific session, ensuring that operations are serialized 
    # and the queue limit is enforced. This method will register the waiting operation, 
    # acquire the session lock, execute the provided Runner function, and finally recycle
    # the waiting count and clean up the session state if no more operations are pending.
    async def Run(self, SessionID, Runner):
        async with self.Guard:
            # Retrieve or create the state for the specified session ID, 
            # including its lock and waiting count.
            State = self.States.setdefault(
                SessionID, { "lock": asyncio.Lock(), "waiting": 0, "last": None }
            )
            # Check if the waiting count for the session has reached the maximum size limit.
            if State["waiting"] >= self.MaxSize:
                # If the queue is full, raise a RuntimeError to indicate that the session's queue is full.
                raise RuntimeError(f"Session {SessionID} queue full (max {self.MaxSize})")
            # Increment the waiting count for the session to indicate that a new operation is waiting.
            State["waiting"] += 1

        try:
            # Acquire the session lock to ensure that operations for this session are executed sequentially.
            async with State['lock']:
                # Execute the provided Runner function, passing the session state, and return its result.
                return await Runner(State)
        finally:
            # Regardless of whether the operation succeeded or failed, decrement the waiting 
            # count for the session within the guard to ensure thread-safe access to the session states.
            async with self.Guard:
                # Decrement the waiting count for the session to indicate that an operation has completed.
                State["waiting"] -= 1
                # If there are no more waiting operations and the session lock is not currently held,
                # remove the session state from the States dictionary to clean up resources.
                if State["waiting"] == 0 and (not State["lock"].locked()):
                    self.States.pop(SessionID, None)

    # Clear the state associated with a specific session ID, removing it from the States dictionary.
    def Clear(self, SessionID):
        # Remove the session state for the specified session ID, if it exists.
        self.States.pop(SessionID, None)

# Register a global instance of the ResponseSessionQueue to manage session operations across the server.
SESSION_QUEUE = ResponseSessionQueue()
    
# Detect whether the input of the responses request contains any runnable content,
# such as non-empty text messages from user or tool outputs. This function is used to 
# determine if there is meaningful input that requires processing, allowing the server 
# to handle requests appropriately based on the presence of runnable input.
def _DetectRunnableInput(Value):
    # For string input, check if it is non-empty after stripping whitespace.
    if isinstance(Value, str):
        return bool(Value.strip())
        
    # Iterate through the list of messages or tool outputs, checking for any non-empty content.
    for Frame in Value or []:
        # For tool output frames, it is considered runnable as long as 
        # the output is not empty after stripping whitespace.
        if Frame.get("type") == "function_call_output" and str(Frame.get("output", "")).strip():
            return True
        # For message frames, check if the content contains any non-empty text, image URLs, 
        # file IDs, file data, or file URLs. If any of these are present, it indicates that 
        # there is runnable input.
        if Frame.get("type") in ("message", None):
            Content = Frame.get("content", "")
            # If the content is a non-empty string, return True.
            if isinstance(Content, str) and Content.strip():
                return True
            # If the content is a list of message parts, check each part for non-empty text, image URLs,
            # file IDs, file data, or file URLs. If any of these are present, return True.
            for Part in Content or []:
                if ( 
                    str(Part.get("text", "")).strip() or
                    Part.get("image_url") or Part.get("file_id") or
                    Part.get("file_data") or Part.get("file_url")
                ):
                    return True

    # If no runnable input is detected in the provided value, return False to 
    # indicate that there is no meaningful input to process.
    return False

'''
WebSocketSessionManager Class
This class is responsible for managing a WebSocket session, handling incoming 
responsive.create messages, and maintaining session state. It provides methods 
to send error messages, manage the overall lifecycle of the WebSocket connection,
and ensure that requests are processed in a controlled manner. The class will 
stream back response events to the client and clean up the session state when the 
connection is closed. It also ensures that requests are processed in a serialized 
manner, preventing race conditions and maintaining the integrity of session data.
'''
class WebSocketSessionManager:
    # Initialize the WebSocketSessionManager with a WebSocket connection and a client instance.
    def __init__(self, WebSocket, Client):
        # Store the WebSocket connection and client instance for use in handling requests and sending responses.
        self.WebSocket    = WebSocket
        self.Client       = Client
        # Initialize session-related attributes to manage the state of the WebSocket session.
        self.SessionID          = None
        self.Instructions       = None
        self.LastInstructionMID = -5201314
        self.Closed             = False

    # Send an error message to the client over the WebSocket connection, 
    # including an optional HTTP status code.
    async def SendErrorMessage(self, Message, Status = 500):
        # Check if the WebSocket connection is still open before attempting to send the error message.
        if not self.WebSocket.closed:
            # Send a JSON-formatted error message to the client
            await self.WebSocket.send_json({
                "type"  : "error",
                "error" : Message,
                "status": Status
            })

    # Process incoming WebSocket messages: response.create requests
    async def ProcessCreate(self, Data):
        try:
            # Attempt to parse the incoming data as JSON
            Message = json.loads(Data)
            # We only accept messages of type response.create for processing,
            # which indicates that the client is requesting the creation of a new response.
            if Message.get("type") != "response.create":
                # If the message type is not response.create, send an error message back to the client.
                return await self.SendErrorMessage("Unknown message type", 400)

            # Extract the relevant data from the message, prioritizing different keys to find the response content.
            RawData = (
                Message.get("response") or Message.get("data") or 
                Message.get("params")   or Message.get("body") or 
                { Key: Value for Key, Value in Message.items() 
                    if Key not in ("type", "requestId", "event_id") }
            )
            # Call the HandleRequest method to process the extracted data, 
            # indicating that it is a Codex request.
            await self.HandleRequest(dict(RawData), True)

        except Exception as Error:
            # If an exception occurs during parsing or processing, send an error message back to the client.
            await self.SendErrorMessage(f"Failed to process message: {Error}", 400)

    # This method is responsible for normalizing and executing a WebSocket response request, 
    # preserving session state. It validates the input, checks for previous response IDs, 
    # normalizes the request data, uploads any necessary files, and executes the request 
    # while streaming back response events to the client. It also handles errors and manages 
    # session state appropriately.
    async def HandleRequest(self, RawData, Codex = False):
        try:
            # Check if the input contains any runnable content, 
            # such as non-empty user messages or function call outputs.
            if not _DetectRunnableInput(RawData.get("input")):
                if Codex:
                    # If the request is from Codex and there is no runnable input, send a response indicating 
                    # that the request was created but completed with an empty response. This allows Codex to 
                    # handle cases where there is no meaningful input without raising an error.
                    await self.WebSocket.send_json({
                        "type"    : "response.created", 
                        "response": {}
                    })
                    await self.WebSocket.send_json({
                        "type"    : "response.completed", 
                        "response": { "id": READY_RESPONSE_ID }
                    })
                    # Record the session ID as READY_RESPONSE_ID to indicate that the session 
                    # is in a ready state with no meaningful input.
                    self.SessionID = READY_RESPONSE_ID
                    return

                # If the request is not from Codex and there is no runnable input, 
                # send an error message back to the client
                return await self.SendErrorMessage(
                    "input must include at least one non-empty user message or function call output.", 
                    400
                )
                
            # Parse the previous response ID from the request data to extract the session ID and 
            # the parent message ID. This is used to maintain the session state and ensure that 
            # the request is associated with the correct session and parent message.
            PreviousResponseID = RawData.get("previous_response_id") or ""
            ParsedSessionID, ParsedMessageID = ParseResponseID(PreviousResponseID)
            # Check if the parsed session ID matches the current session ID. If they do not match,
            # send an error message back to the client indicating that the previous response was not found.
            if ParsedSessionID and self.SessionID and (ParsedSessionID != self.SessionID):
                return await self.SendErrorMessage({
                    "code"   : "previous_response_not_found", 
                    "message": f"Previous response with id '{ParsedSessionID}' not found, expected {self.SessionID}.", 
                    "param"  : "previous_response_id"
                }, 400)

            # Process the instructions in the request data.
            if RawData.get("instructions") is not None:
                # If the instructions are the same as the last instructions and the message ID is within 
                # 30 of the last instruction message ID, remove the instructions from the request data to 
                # avoid redundant processing. This helps to prevent unnecessary re-execution of the same 
                # instructions when they have not changed significantly.
                if RawData.get("instructions") == self.Instructions and (ParsedMessageID or 0) - self.LastInstructionMID < 30:
                    RawData.pop("instructions", None)
                else:
                    # Otherwise, update the last instructions with the new instructions from the request data.
                    self.Instructions = RawData.get("instructions")

            # Normalize the request data to ensure that it is in a consistent format for processing.
            try:
                ResInfo, Choice = NormalizeResponses(RawData)
            except ValueError as Error:
                # If a ValueError occurs during normalization, check if it is due to an unsupported model. 
                # If so, change the model to "deepseek" and attempt normalization again. Otherwise, raise 
                # the error to be handled by the outer exception handler.
                if Codex and "Unsupported model:" in str(Error):
                    # Fall back to the default model if the specified model is unsupported, 
                    # allowing the request to be processed with a known working model.
                    RawData["model"] = "deepseek"
                    # Re-attempt normalization with the default model
                    ResInfo, Choice = NormalizeResponses(RawData)
                else:
                    # If the error is not related to an unsupported model, raise the error 
                    # to be handled by the outer exception handler.
                    raise Error

            # Upload any necessary files from the request data to the backend client, 
            # obtaining their corresponding file IDs.
            FileIDs          = await UploadAttachmentsResponses(RawData, self.Client, ResInfo["model_type"])
            # Obtain the session ID from the request data, falling back to the current session ID if not provided.
            RequestSessionID = ResInfo.get("session_id") or self.SessionID
            # Determine the session ID to use for queuing the request. The placeholder session should not
            # be added to the queue, as it is used to indicate that there is no meaningful input to process.
            if RequestSessionID and (not RequestSessionID.startswith(READY_RESPONSE_ID)):
                QueueSessionID = RequestSessionID
            else:
                QueueSessionID = None

            # EXECUTE MODULE: Asynchronously execute the request within the session queue, 
            # ensuring that requests for the same session are serialized.
            async def Execute(State = None):
                # Obtain the parent message ID from the state if available, otherwise use the parsed message ID.
                if State and State.get("last") is not None:
                    ParentMessageID = State.get("last")
                else:
                    ParentMessageID = ResInfo.get("parent_message_id")
                # Obtain the session ID to use for the request, falling back to the current session ID if not provided.
                if RequestSessionID and RequestSessionID.startswith(READY_RESPONSE_ID):
                    RequestSid = None
                else:
                    RequestSid = RequestSessionID

                # Invoke the backend client's chat completion method
                SessionID, ResponseBody = await self.Client.ChatCompletion({
                    **ResInfo,
                    "session_id"        : RequestSid,
                    "parent_message_id" : ParentMessageID,
                    "file_ids"          : FileIDs
                })
                # Update the current session ID with the session ID returned from the backend client.
                self.SessionID = SessionID

                # Initialize a variable to hold the last response event received from the backend client.
                LastEvent = None

                # Stream response events from the backend client and send them to the WebSocket client.
                async for Event in ResponseEvents(
                    ResponseBody, SessionID, ResInfo["model"], len(ResInfo["message"]),
                    WhethertoParseToolCall(Choice, RawData.get("tools"))
                ):
                    # Check if the WebSocket connection is still open before attempting to send the event.
                    if not self.WebSocket.closed:
                        # Send the event as a JSON-formatted message to the WebSocket client.
                        await self.WebSocket.send_json(Event)
                    # If the event indicates that the response is completed, store it as the last event.
                    if Event.get("type") == "response.completed":
                        LastEvent = Event

                # After streaming all events, check if there was a completed response event received.
                if LastEvent:
                    # If there was a completed response event, parse its response ID to obtain the message ID.
                    _, MessageID = ParseResponseID(LastEvent['response']['id'])
                    # Update the last instruction message ID with the parsed message ID, 
                    # or use a default value if not available.
                    self.LastInstructionMID = MessageID if MessageID is not None else -5201314
                    # If the state is available, update its last message ID with the parsed message ID.
                    if State is not None:
                        State["last"] = MessageID

            # Run the Execute function within the session queue, 
            # ensuring that requests for the same session are serialized.
            if QueueSessionID:
                await SESSION_QUEUE.Run(QueueSessionID, Execute)
            else:
                await Execute()

        except Exception as Error:
            # If an exception occurs during request handling, send an error message back to the client.
            if Codex and (not self.WebSocket.closed):
                await self.WebSocket.send_json({
                    "type"      : "response.failed", 
                    "response"  : {
                        "status": "failed", 
                        "error" : {
                            "type"   : "server_error", 
                            "code"   : "server_error", 
                            "message": str(Error)
                        }
                    }
                })

            # If the exception is caused by an unsupported model, send a 400 Bad Request error. 
            # Otherwise, send a 500 Internal Server Error.
            if "Unsupported model:" in str(Error):
                await self.SendErrorMessage(str(Error), 400)
            else:
                await self.SendErrorMessage(str(Error), 500)

    # Clean up the WebSocket session by releasing the active session and 
    # removing its queued state on connection close.
    async def Cleanup(self):
        # If the session has already been closed, return early to avoid redundant cleanup.
        if self.Closed:
            return

        # Mark the session as closed to prevent further processing of requests.
        self.Closed = True
        # Obtain the current session ID to be released.
        SessionID   = self.SessionID
        # If there is a valid session ID and it does not start with the READY_RESPONSE_ID
        if SessionID and (not SessionID.startswith(READY_RESPONSE_ID)):
            try:
                # Attempt to delete the session from the backend client, releasing any associated resources.
                await self.Client.DeleteSession(SessionID)
            except Exception:
                # If an exception occurs during session deletion, ignore it to avoid disrupting the cleanup process.
                pass
            # Clear the session state from the SESSION_QUEUE to remove any queued operations for this session.
            SESSION_QUEUE.Clear(SessionID)

'''
App Class
This class implements the main application logic for handling OpenAI-compatible API requests. 
It provides methods for processing model listings, chat completions, and responses, including 
support for streaming and non-streaming responses. The class interacts with the backend client 
to execute requests, manage sessions, and handle file uploads. It also includes error handling 
and session lifecycle management to ensure that requests are processed correctly and efficiently.
'''
class App:
    # Initialize the App with a backend client instance, 
    # which will be used to handle requests and manage sessions.
    def __init__(self, Client):
        self.Client        = Client
        # Initialize a dictionary to hold locks for each session, 
        # ensuring that requests for the same session are serialized.
        self.Locks         = {}
        # Initialize a dictionary to hold the last message ID for each session,
        # allowing the application to track the state of each session and maintain continuity between requests.
        self.LastMessageID = {}

    # GET /v1/models
    # Return the list of supported models exposed by the local compatibility endpoint in a JSON response.
    async def Models(self, Request):
        # Return a JSON response containing the list of supported models, 
        # including their IDs, object types, creation timestamps, and ownership information.
        return web.json_response({
            "object": "list",
            "data"  : [
                {
                    "id"        : ModelID,
                    "object"    : "model",
                    "created"   : 0,
                    "owned_by"  : "localhost"
                } for ModelID in MODELS_LIST
            ]
        })

    # POST /v1/chat/completions
    # Process a chat completion request, supporting both streaming and non-streaming responses.
    # The method normalizes the request data, uploads any necessary files, and invokes the backend client's 
    # chat completion method. It then streams or packages the results and returns them to the client.
    async def ChatCompletions(self, Request):
        # Parse the incoming request data as JSON and normalize it for processing.
        RawData                = await Request.json()
        NormalizedData, Choice = NormalizeCompletions(RawData)

        # If the normalized message is empty, return a 400 Bad Request error 
        # indicating that at least one non-empty message is required.
        if not NormalizedData["message"].strip():
            return web.json_response({
                "error": {
                    "message": "messages must include at least one non-empty message.",
                    "type"   : "invalid_request_error"
                }
            }, status = 400)

        # Upload any necessary files from the request data to the backend client, 
        # obtaining their corresponding file IDs.
        FileIDs   = await UploadAttachmentsCompletions(RawData, self.Client, NormalizedData["model_type"])
        # Initialize the session ID to None, which will be used to track the session state for the request.
        SessionID = None

        try:
            # Initiate the chat completion request to the backend client, passing the normalized data,
            # session ID, and file IDs. The backend client will return a session ID and a response body stream.
            SessionID, ResponseBody = await self.Client.ChatCompletion({
                **NormalizedData, 
                "session_id": SessionID, 
                "file_ids"  : FileIDs
            })

            # For streaming responses, prepare a streaming response with appropriate headers and stream the
            # response events from the backend client to the client. Each event is sent as a JSON-formatted 
            # message, and a final [Done] message will be sent to indicate the end of the stream.
            if RawData.get('stream'):
                # Prepare a streaming response with appropriate headers for Server-Sent Events (SSE).
                SResponse = web.StreamResponse(headers = {
                    "Content-Type"                : "text/event-stream; charset=utf-8",
                    "Cache-Control"               : "no-cache, no-transform",
                    "Connection"                  : "keep-alive",
                    "Access-Control-Allow-Origin" : "*",
                    "Access-Control-Allow-Headers": "content-type, authorization, x-requested-with",
                    "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
                })
                # Prepare the streaming response to be sent to the client.
                await SResponse.prepare(Request)
                # Stream response events from the backend client and send them to the client as JSON-formatted messages.
                async for Chunk in CompletionEvents(
                    ResponseBody, SessionID, NormalizedData["model"], len(NormalizedData["message"]),
                    WhethertoParseToolCall(Choice, RawData.get("tools"))
                ):
                    # Send each chunk of the response as a JSON-formatted message to the client.
                    await SResponse.write(f"data: {json.dumps(Chunk, ensure_ascii = False)}\n\n".encode())

                # After streaming all chunks, send a final [Done] message to indicate the end of the stream.
                await SResponse.write(b"data: [DONE]\n\n")
                # Return the streaming response to the client.
                return SResponse

            # For non-streaming responses, parse the response body to extract the final result,
            # including the text, reasoning content, and token usage. Package the result into a JSON
            # response and return it to the client.
            Parsed    = await ParseStream(ResponseBody)
            # Determine whether to parse tool calls based on the choice and the tools provided in the raw data.
            ToolMatch = WhethertoParseToolCall(Choice, RawData.get("tools"))
            Message   = InternalToCompletions(Parsed["text"], ToolMatch)
            # If there is any reasoning content in the parsed response, include it in the message.
            if Parsed["thinking"].strip():
                Message["reasoning_content"] = Parsed["thinking"].strip()
            # Estimate the token usage based on the accumulated token usage, input length, and output length.
            TokenUsage = EstimateTokenUsage(
                Parsed["accumulated_token_usage"], 
                len(NormalizedData["message"]), 
                len(Parsed["text"]) + len(Parsed["thinking"])
            )

            # Construct the final result to be returned to the client, 
            # including the response ID, model, choices, and token usage.
            Result = {
                "id"      : SessionID,
                "object"  : "chat.completion",
                "created" : int(time.time()),
                "model"   : NormalizedData["model"],
                "choices" : [{
                    "index"         : 0,
                    "message"       : Message,
                    "finish_reason" : "tool_calls" if Message.get("tool_calls") else "stop"
                }],
                "usage"   : {
                    "total_tokens"      : TokenUsage["total_tokens"],
                    "prompt_tokens"     : TokenUsage["input_tokens"],
                    "completion_tokens" : TokenUsage["output_tokens"]
                }
            }
            # Return the final result as a JSON response to the client.
            return web.json_response(Result)

        finally:
            # After processing the request, attempt to delete the session from the backend client to
            # release any associated resources. If an exception occurs during session deletion, ignore it.
            try:
                await self.Client.DeleteSession(SessionID)
            except Exception:
                pass

# Create the aiohttp web application, configure middleware, endpoints, and application cleanup.
# This function initializes the web application, sets up the necessary middleware for IP 
# allowlisting and CORS, defines the endpoints for health checks, model listings, chat completions, 
# and responses, and registers a cleanup function to close the backend client when the application 
# is shutting down. The application is returned to be run by the aiohttp web server. 
async def CreateApp(Client):
    # Create an instance of the App class with the provided backend client, which will handle
    # the logic for processing requests and managing sessions.
    AppInstance = App(Client)
    # Create the aiohttp web application with the specified middleware for IP allowlisting and CORS.
    Application = web.Application(middlewares = [IPWhiteListMiddleware, CORSMiddleware])

    # OPTIONS PRE CHECK HANDLER
    # This handler responds to OPTIONS requests, which are typically sent by browsers as 
    # part of the CORS preflight process. It returns a 204 No Content response with the 
    # appropriate CORS headers to allow cross-origin requests from the client.
    async def OptionsHandler(Request):
        # Return a 204 No Content response with the appropriate CORS headers to allow cross-origin requests.
        return web.Response(
            status  = 204,
            headers = {
                "Access-Control-Allow-Origin" : "*",
                "Access-Control-Allow-Headers": "content-type, authorization, x-requested-with",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
            }
        )

    # HEALTH CHECK HANDLER
    # This handler responds to health check requests, indicating that the server is functioning 
    # correctly. It returns a JSON response with a simple "ok" message. This is useful for 
    # monitoring the health of the application. It can be called by external tools or services 
    # to verify that the server is running and responsive.
    async def HealthHandler(Request):
        return web.json_response({ "ok": True })

    # COMPLETIONS HANDLER
    # This handler processes chat completion requests sent to the /v1/chat/completions endpoint.
    # It delegates the request to the AppInstance's ChatCompletions method, which handles the
    # logic for processing the request, including normalization, file uploads, and invoking the
    # backend client's chat completion method. The response is then returned to the client.
    async def CompletionsHandler(Request):
        return await ErrorHandler(Request, AppInstance.ChatCompletions)

    # APPLICATION CLEANUP HOOKS
    # This function is registered as a cleanup hook for the application, which will be called
    # when the application is shutting down. It ensures that the backend client is properly closed,
    # releasing any resources and connections associated with it. This is important for preventing
    # resource leaks and ensuring that the application shuts down cleanly.
    async def CleanupClient(_App):
        await Client.Close()

    # Next is the routing registration for the application, 
    # which defines the endpoints and their corresponding handlers.
    Application.router.add_route("OPTIONS", "/{tail:.*}", OptionsHandler)

    Application.router.add_get("/health"   , HealthHandler)
    Application.router.add_get("/v1/models", AppInstance.Models)
   
    Application.router.add_post("/v1/chat/completions", CompletionsHandler)

    # Register the cleanup function to be called when the application is shutting down,
    # ensuring that the backend client is properly closed and resources are released.
    Application.on_cleanup.append(CleanupClient)

    return Application

# Main entry point for the application. Parse command-line arguments, create 
# the backend client, and start the aiohttp web server. The server will listen for 
# incoming requests and handle them using the defined application logic. 
async def _Main():
    # Create an argument parser
    Parser = argparse.ArgumentParser()
    # Add command-line arguments for:
    # Port: The port number on which the server will listen for incoming requests.
    Parser.add_argument("-p", "--port", type = int, default = 10086, 
                        help = "Port number for the server to listen on (Default: 10086)")
    # Credentials: The path to the credentials file for the backend client.
    Parser.add_argument("-c", "--credentials", default = DefaultCredentialPath(),
                        help = "Path to the credentials file for the backend client")
    # Browser: An optional flag indicating whether to use a browser for interaction.
    Parser.add_argument("--browser", action = "store_true", help = "Use a browser for interaction")
    # User Data Directory: An optional argument specifying the user data directory for the application.
    Parser.add_argument("--user-data-dir", help = "User data directory for the application")

    Arguments = Parser.parse_args()
    
    # Create the backend client
    Client = await CreateServerClient(Arguments.credentials, Arguments.browser, Arguments.user_data_dir)
    # Create the aiohttp web application with the backend client
    App    = await CreateApp(Client)
    Runner = web.AppRunner(App)
    await Runner.setup()
    # Start the web server and listen on the specified port
    Site   = web.TCPSite(Runner, "127.0.0.1", Arguments.port)
    await Site.start()

    # Print a message indicating that the HTTP server is listening on the specified address and port.
    print(f"FreeAPI > HTTP server listening on http://127.0.0.1:{Arguments.port}")

    await asyncio.Event().wait()

# Synchronous entry point for callers of the command-line module.
def main():
    asyncio.run(_Main())

# Backward-compatible aliases preserve potential external Python call sites.
_main = _Main
