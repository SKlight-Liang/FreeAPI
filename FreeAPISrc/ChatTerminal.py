'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module provides a command-line interface to conduct interactive conversations 
with the DeepSeek web-based chat client. It will use local credentials to disguise
itself as a normal user. In this way, the module can create chat sessions, submit
messages and parse the responses in a streaming manner. The reasoning text will be
sent to stderr, while the response content will be sent to stdout. This is a real-time
processing interface, so the user can see the response as it is being generated.
We also support an interactive mode, where the user can continue chatting 
after the initial message.
'''

import sys
import asyncio
import argparse

from datetime import datetime

from .Utils import *
from .DeepSeekAuth import *
from .DeepSeekWebClient import *
from .DeepSeekStreamParser import *

# DEBUG: We define a debugging function to log the raw SSE events to help 
# us find some potential patterns in the streaming responses. This is useful 
# for us to improve the parsing logic and understand the underlying protocol.
def DebugEventLogger(Result, Events):
    # Create a folder to store the raw SSE events. 
    # The folder will be created in the default user data directory.
    Directory = pathlib.Path(DefaultDataDir(), "Log")
    Directory.mkdir(parents = True, exist_ok = True)

    # Generate a timestamped filename
    Timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    FilePath  = Directory / f"DeepSeekFragments [{Timestamp}].json"

    # Build a JSON object to store everything we want to log.
    Data = {
        "message_id"         : Result.get("message_id"),
        "fragments"          : Result.get("fragments", []),
        "references"         : Result.get("references", []),
        "reference_registry" : Result.get("reference_registry", {}),
        "events"             : Events
    }

    # Write the JSON object to the file in a human-readable format with indentation.
    with FilePath.open("w", encoding = "utf-8") as File:
        json.dump(Data, File, ensure_ascii = False, indent = 2)

    print(f"FreeAPI > Debug log saved to {FilePath}", file = sys.stderr)

# Resolve command-line arguments and run the chat workflow. 
# We will load configuration from the local path to establish 
# client and sessions. It realizes loop conversations with the model
# and streaming responses in an asynchronous manner.
async def _Main():
    # Create an argument parser
    Parser = argparse.ArgumentParser()
    # Add command-line arguments for:
    # Message: an optional initial message to send to the chat model.
    Parser.add_argument("message", nargs = "?")
    # Credentials: an optional path to the local credentials file.
    Parser.add_argument("-c", "--credentials", default = DefaultCredentialPath())
    # Interactive: a flag to enable interactive mode, 
    # where the user can continue chatting after the initial message.
    Parser.add_argument("-i", "--interactive", action = "store_true", 
                        help = "Continue chatting after the initial message")
    # Delete: a flag to delete the chat session after completion.
    Parser.add_argument("-d", "--delete", action = "store_true",
                        help = "Delete the chat session after completion")
    # Tool: a flag to show tool snapshots in the output.
    Parser.add_argument("-t", "--tool", action = "store_true",
                        help = "Show tool snapshots in the output")
    # Search Detail: a flag to enable detailed search results in the tool snapshots.
    Parser.add_argument("-s", "--search-detail", action = "store_true",
                        help = "Enable detailed search results in tool snapshots")
    # Debug: a flag to enable debug mode, which will print additional information.
    Parser.add_argument("--debug", action = "store_true",
                        help = "Enable debug mode")

    Arguments = Parser.parse_args()

    # Create a DeepSeekWebClient instance to handle chat interactions
    # with the provided credentials and debug settings.
    Client    = DeepSeekWebClient(
        LoadCredentials(Arguments.credentials),
        Verbose = Arguments.debug
    )
    # Initialize session and parent message identifiers to None for the first request.
    SessionID       = None
    ParentMessageID = None

    try:
        # Create a new chat session for the conversation. This session will be 
        # shared across all messages in this invocation. Obtain the session ID.
        SessionID = await Client.CreateChatSession()

        # OPTIONAL: If debug mode is enabled, print the session creation
        # information such as the session ID to stderr for diagnostic purposes.
        if Arguments.debug:
            print(f"FreeAPI > Chat session created: {SessionID}", file = sys.stderr)

        # Retrieve the first message from the command-line arguments
        Message = Arguments.message

        # Loop to handle multiple rounds of chat if there are messages or the 
        # interactive mode is enabled. In one-shot mode, we will submit at most
        # one message; while in interactive mode, we will continue prompting 
        # after each successfully parsed response.
        while Message or Arguments.interactive:
            # If there is no ready-made message, read a line from standard input 
            # and remove the leading and trailing blanks. 
            if not Message:
                Message = input("YOU > ").strip()
                # If the input is empty, continue to the next iteration and wait for input again.
                if not Message:
                    continue

            # If the message entered by the user is "exit", break the loop.
            if Message.lower() == "exit":
                break

            # Initialize the stream state of incremental output. This flag will 
            # be used to determine when the stream changes between content kinds.
            StreamState = None

            # BUG: Initialize a dictionary to store the displayed tool snapshots.
            # This is used to avoid printing duplicate tool snapshots in the output. 
            # Each tool snapshot is identified by its unique ID.
            DisplayedTools = {}

            # STREAMING CALLBACK:
            # Real-time output of the reasoning or response content to the corresponding 
            # output streams (stdout for response content, stderr for reasoning). The 
            # callback will be invoked for each delta received from the streaming response. 
            # If the stream changes between content kinds, we will output some prompting 
            # information to indicate the change. This is friendly for users to distinguish 
            # between reasoning and response content.
            def DeltaCallbackHandler(Kind, Text):
                # Modify the StreamState variable from the outer scope.
                nonlocal StreamState
                # If the stream kind changes, print prompt to separate the different content.
                if Kind != StreamState:
                    if Kind == "THINK":
                        # If the previous stream was not a tool stream, print a newline.
                        if StreamState is not None and StreamState != "TOOL":
                            print(flush = True, file = sys.stderr)
                        print("THINKING >", flush = True, file = sys.stderr)
                    elif Kind == "RESPONSE":
                        if StreamState is not None and StreamState != "TOOL":
                            print(flush = True)
                        print("REPLY >", flush = True)

                # Update the StreamState to the current kind for the next delta.
                StreamState = Kind

                # Print the delta text to the appropriate output stream based on its kind.
                if Kind == "THINK":
                    print(Text, end = "", flush = True, file = sys.stderr)
                elif Kind == "RESPONSE":
                    print(Text, end = "", flush = True)

            # FEATURE: We add a OnTool callback to output the tool snapshots in real-time.
            # We hope to get a better understanding through the tool usage and the 
            # intermediate reasoning process. We believe that there must be a lot of people
            # being insterested in the working principle behind the model.
            def ToolCallbackHandler(Snapshot):
                nonlocal StreamState

                if StreamState is not None and StreamState != "TOOL":
                    print(flush = True, file = sys.stderr)
                # Update the StreamState to "TOOL" for the next delta.
                StreamState = "TOOL"

                # Retrieve the type and status of the tool call
                Kind   = Snapshot.get("type"  , "UNKNOWN")
                Status = Snapshot.get("status", "UNKNOWN")

                if Kind == "TOOL_SEARCH":
                    # BUG: We will only print the tool snapshot if it has not been printed before.
                    ToolID = Snapshot.get("id")
                    DisplayState = DisplayedTools.setdefault(ToolID, {
                        "queries": set(),
                        "results": set(),
                        "status" : None
                    })

                    # If the status of the tool call has changed, print the new status to stderr.
                    if DisplayState["status"] != Status:
                        print(f"TOOL SEARCH > Status: {Status}", flush = True, file = sys.stderr)

                        DisplayState["status"] = Status

                    # Print the query list of the tool call.
                    for Item in Snapshot.get("queries", []):
                        Query = Item.get("query")

                        # If the query has not been printed before, 
                        # print it to stderr and add it to the displayed set.
                        if Query not in DisplayState["queries"]:
                            print(f"  QUERY: {Query}", flush = True, file = sys.stderr)

                            DisplayState["queries"].add(Query)

                    # If the tool call is finished, print the result list of the tool call.
                    # This is because a tool call may trigger multiple callbacks.
                    # We only want to print the results when the tool call is finished.
                    if Status == "FINISHED":
                        for Item in Snapshot.get("results", []):
                            URL      = Item.get("url")
                            Title    = Item.get("title")
                            Abstract = Item.get("snippet")

                            # If the result has not been printed before,
                            # print it to stderr and add it to the displayed set.
                            if URL not in DisplayState["results"]:
                                print(f"  RESULT: {Title}"   , flush = True, file = sys.stderr)
                                print(f"          {URL}"     , flush = True, file = sys.stderr)
                                if Arguments.search_detail:
                                    print(f"          {Abstract}", flush = True, file = sys.stderr)

                                DisplayState["results"].add(URL)

                elif Kind == "TOOL_OPEN":
                    if Status == "FINISHED":
                        print(
                            f"TOOL OPEN > URL: {(Snapshot.get("result") or {}).get('url')}", 
                            flush = True, file = sys.stderr
                        )

                else:
                    print(f"TOOL {Kind} > Status: {Status}", flush = True, file = sys.stderr)

            # DEBUG: If debug mode is enabled, we will log the raw SSE events to a file for analysis.
            EventHistory = []
            # We will define a callback to log the raw SSE events for debugging purposes.
            def EventCallbackHandler(Event):
                EventHistory.append(copy.deepcopy(Event))

            # Send a chat completion request with the current session, parent message 
            # identifiers and the message text. The response will be streamed back, 
            # and the DeltaCallbackHandler will be invoked for each delta received.
            ResponseBody = await Client.ChatCompletions({
                "session_id"       : SessionID,
                "parent_message_id": ParentMessageID,
                "message"          : Message
            })
            # Analyze the streamed response to extract the final result. 
            # Use delta callback to output incremental content in real-time.
            Result       = await ParseStream(
                ResponseBody, 
                OnDelta = DeltaCallbackHandler,
                OnTool  = ToolCallbackHandler   if Arguments.tool  else None,
                OnEvent = EventCallbackHandler  if Arguments.debug else None
            )

            # After the streaming is complete, we will print a newline to separate 
            # the final output. Of course, this is based on the incremental output 
            # already printed by the callback.
            if StreamState is not None:
                print(flush = True)
            # Otherwise, if the response contains text, print it to stdout.
            # This situation usually occurs in scenarios without incremental callbacks.
            elif Result["text"]:
                print(Result["text"], flush = True)
            # If the program reaches this point, it indicates that the flow has ended 
            # without a formal response. In this case, we need to raise an exception.
            else:
                raise RuntimeError((
                    "DeepSeek completion stream ended without RESPONSE content. " 
                    "Run with --debug for request diagnostics."
                ))

            # FEATURE: Print resolved citation URLs after the completed response.
            References = Result.get("references", {})

            # DEBUG: Log the raw SSE events to a file for analysis.
            if Arguments.debug:
                DebugEventLogger(Result, EventHistory)

            # DEBUG PART: If debug mode is enabled, print the resolved references to stderr.
            if Arguments.debug:
                for Reference in Result.get("references", []):
                    print(f"FreeAPI > Reference Information :"      , file = sys.stderr)
                    print(f"Index      = {Reference.get('index')}"  , file = sys.stderr)
                    print(f"Ordinal    = {Reference.get('ordinal')}", file = sys.stderr)
                    print(f"ResponseID = {Reference.get('response_fragment_id')}", file = sys.stderr)
                    print(f"ToolID     = {Reference.get('tool_fragment_id')}"    , file = sys.stderr)
                    print(f"ToolType   = {Reference.get('type')}"   , file = sys.stderr)
                    print(f"URL        = {Reference.get('url')}"    , file = sys.stderr)

            if References:
                print("CITATIONS >", flush = True, file = sys.stderr)
                for Reference in References:
                    # Retrieve the URL and title of the reference
                    Index = Reference.get("index")
                    URL   = Reference.get("url") or "[Unresolved]"
                    Title = Reference.get("title") or "[No Title]"
                    # Print the citation information to stderr.
                    print(f"  [Reference:{Index}] {Title}", flush = True, file = sys.stderr)
                    print(f"    URL: {URL}", flush = True, file = sys.stderr)
            
            # Update the parent message ID for the next request to maintain the conversation context.
            ParentMessageID = Result["message_id"]

            # OPTIONAL: If debug mode is enabled, print the message ID and token usage to stderr.
            if Arguments.debug:
                print((
                    f"FreeAPI > response MessageID = {ParentMessageID}, "
                    f"Token Usage = {Result['accumulated_token_usage']}"
                ), file = sys.stderr)

            # Reset the message for the next iteration.
            Message = None

            # In non interactive mode, we will break the loop after the first message is processed.
            if not Arguments.interactive:
                break

    # Clean up resources and close the client after the chat session is complete.
    finally:
        # If the delete flag is set and a session was created, delete the session.
        if Arguments.delete and SessionID:
            await Client.DeleteSession(SessionID)
        # Close the client to release any underlying resources.
        await Client.Close()
                
# Synchronous entry point for callers of the command-line module.
def main():
    asyncio.run(_Main())

# Backward-compatible aliases preserve potential external Python call sites.
_main = _Main            
