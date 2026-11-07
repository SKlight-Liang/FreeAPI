'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module provides some utilities for parsing and handling Deepseek API responses, 
especially those received in a streaming manner. This is because we are actually interacting with the web side.
The DeepSeekStreamParser class is designed to handle Server-Sent Events (SSE) streams,
which are commonly used for real-time updates in web applications. 
It can progressively receive data chunks and parse out complete SSE events, 
maintaining the state of the response as it arrives.
The DeepSeekStateDecoder class is used internally by DeepSeekStreamParser to manage the state of the response,
including the message content, title, session updates, and any hints or errors that may be sent by the server. 

Functions Table:
- ParseStream(Body, OnDelta = None):
An asynchronous function that reads data from the provided stream and parses it.
'''

from .SSE import SSEParser

'''
DeepSeekStateDecoder Class
This class is responsible for maintaining the status in the resolution process, 
handling SSE events, supporting path settings, incremental updates, batch operations, etc., 
and outputting incremental content through OnDelta callback.

Functions Table(Public):
- _NormPath(Path):
Normalizes the path to ensure it starts with "message" and is in the correct format.
- _ConvIndex(QueryList, Token, Writable=False):
Converts a token to an index for list access, handling negative indices and bounds checking.
- GetValue(Tokens):
Retrieves the value at the specified path of tokens, returning None if the path does not exist.
- SetValue(Tokens, Value, Append=False):
Sets the value at the specified path of tokens, creating intermediate structures as needed.
- _Delta(Tokens, Value):
Handles incremental updates by invoking the OnDelta callback with the type and content of the fragment.
- Apply(Path, Operation, Value):
Applies the specified operation (SET, APPEND, or BATCH) to the value at the given path.
- Push(Item):
Processes an incoming SSE event item, updating the state based on the event type and data.
'''
class DeepSeekStateDecoder:
    # Initialize the decoder, and set the initial state, default path, default operation, and incremental callback.
    def __init__(self, OnDelta = None):
        self.State   = {
            "ready"          : None,
            "title"          : None,
            "close"          : None,
            "update_session" : [],
            "message"        : {},
            "hint"           : None
        }
        self.Path    = "message"
        self.OP      = "SET"
        self.OnDelta = OnDelta

    # Normalize the path to ensure it starts with "message" and is in the correct format.
    def _NormPath(self, Path): 
        # If the path is empty or None, default to "message"
        if not Path or not isinstance(Path, str):
            return "message"
        # If the path already starts with "message", return it as is
        if Path.startswith("message"):
            return Path
        # Otherwise, prepend "message/" to the path
        return f"message/{Path}"
        
    # Helper function to convert a token to an index for list access, handling negative indices and bounds checking.
    def _ConvIndex(self, QueryList, Token , Writable = False):
        # Attempt to convert the token to an integer index
        try:
            Index = int(Token)
        except:
            return None
        
        # If the index is negative, adjust it to be a positive index from the end of the list
        if Index < 0:
            Index += len(QueryList)
        # If Writable is True and the index is still negative, set it to 0 (to allow appending)
        if Writable and Index < 0:
            Index = 0

        # Return the index if it is within bounds, otherwise return None
        if Writable or (0 <= Index < len(QueryList)):
            return Index
        return None
        
    # Get the value at the specified path of tokens, returning None if the path does not exist.
    def GetValue(self, Tokens):
        # Start from the root of the state
        Value = self.State
        # Iterate through the tokens to traverse the nested structure
        for Token in Tokens:
            try:
                # If the current value is a list, convert the token to an index
                if isinstance(Value, list):
                    Value = Value[self._ConvIndex(Value, Token)]
                else:
                    # Otherwise, treat it as a dictionary key
                    Value = Value[Token]
            except:
                # If any access fails, return None
                return None

        return Value
        
    # Set the value at the specified path of tokens, creating intermediate structures as needed.
    # If append is True and the existing value is a string, concatenate the new value to the existing string.
    def SetValue(self, Tokens, Value, Append = False):
        # Start from the root of the state
        Node     = self.State
        NextNode = None
        # Iterate through the tokens, except for the last one, to traverse/create the nested structure
        for i, Token in enumerate(Tokens[:-1]):
            # Determine the next structure to create based on whether the next token is a digit (list) or not (dict)
            if Tokens[i + 1].lstrip('-').isdigit():
                NextNode = []
            else:
                NextNode = {}

            if isinstance(Node, list):
                # If the current node is a list, that means the token should be treated as an index
                Index = self._ConvIndex(Node, Token, True)
                # Ensure that the list is long enough to accommodate the index
                while len(Node) <= Index:
                    Node.append(None)
                # This usually happens when the Node[Index] is just created
                if not isinstance(Node[Index], (dict, list)):
                    Node[Index] = NextNode
                Node = Node[Index]
            else:
                if not isinstance(Node.get(Token), (dict, list)):
                    Node[Token] = NextNode
                Node = Node[Token]

        # Handle the last token to set the value
        LastToken = Tokens[-1]
        if isinstance(Node, list):
            Index = self._ConvIndex(Node, LastToken, True)
            while len(Node) <= Index:
                Node.append(None)
            OldValue = Node[Index]

            # If Append is True and both the old and new values are strings, concatenate them.
            # Otherwise, simply set the new value.
            if Append and isinstance(OldValue, str) and isinstance(Value, str):
                Node[Index] = OldValue + Value
            # BUG: In the actual test, both OldValue and Value are LIST, then old value will be discarded.
            # This lead to the loss of some fragments in the final result such as the thinking content.
            elif Append and isinstance(OldValue, list) and isinstance(Value, list):
                Node[Index].extend(Value)
            else:
                Node[Index] = Value

        else:
            OldValue = Node.get(LastToken)
            if Append and isinstance(OldValue, str) and isinstance(Value, str):
                Node[LastToken] = OldValue + Value
            elif Append and isinstance(OldValue, list) and isinstance(Value, list):
                Node[LastToken].extend(Value)
            else:
                Node[LastToken] = Value

    # Handle incremental updates by invoking the OnDelta callback with the type and content of the fragment.
    def _Delta(self, Tokens, Value):
        # If the OnDelta callback is not set, return directly
        if not self.OnDelta:
            return

        # Define a helper function to emit the callback event for a fragment.
        def Emit(Fragment):
            # If the fragment is a dictionary with a non-empty string content, invoke the OnDelta callback.
            if isinstance(Fragment, dict) and isinstance(Fragment.get("content"), str) and Fragment.get("content"):
                self.OnDelta(str(Fragment.get("type", "")).upper(), Fragment["content"])

        # Handle different cases based on the tokens and value to determine how to emit fragments.
        if Tokens == ["message"]:
            # "Message" event usually contains multiple fragments in its response.
            # So we need to iterate through them and emit each one.
            if isinstance(Value, dict):
                Response  = Value.get("response", {})
                Fragments = Response.get("fragments", [])
            else:
                Fragments = []

            for Fragment in Fragments:
                Emit(Fragment)

        elif Tokens[-1:] == ["fragments"] and isinstance(Value, list):
            # If the last token is "fragments", emit each fragment in the list.
            for Fragment in Value:
                Emit(Fragment)

        elif Tokens[-1:] == ["content"] and isinstance(Value, str):
            # If the last token is "content", emit the fragment with its type and content.
            Fragment = self.GetValue(Tokens[:-1])
            if isinstance(Fragment, dict):
                Emit({
                    "type": Fragment.get("type"),
                    "content": Value
                })
            else:
                Emit({
                    "type": None,
                    "content": Value
                })

    # Apply the specified operation (SET, APPEND, or BATCH) to the value at the given path.
    def Apply(self, Path, Operation, Value):
        # Normalize the path to ensure it starts with "message" and is in the correct format.
        NormPath = self._NormPath(Path)
        # Split the normalized path into tokens for traversal and manipulation.
        Tokens   = [x for x in NormPath.split('/') if x]

        # Handle batch operations by recursively applying each operation in the list.
        if Operation == "BATCH" and isinstance(Value, list):
            for Item in Value:
                if isinstance(Item, dict):
                    SubPath = f"{NormPath}/{Item['p']}" if Item.get('p') else NormPath
                    SubOp   = Item.get('o', "SET")
                    SubVal  = Item.get('v')
                    self.Apply(SubPath, SubOp, SubVal)
            return

        # For non-batch operations, apply the delta and set the value accordingly.
        self._Delta(Tokens, Value)
        self.SetValue(Tokens, Value, Operation == "APPEND")

    # Process an incoming SSE event item, updating the state based on the event type and data.
    def Push(self, Item):
        # Extract the event type and data from the incoming item.
        Event = Item["event"]
        Data  = Item["data"]
    
        if not isinstance(Data, dict):
            return

        # Update the state based on the event type, 
        # handling various cases such as "ready", "update_session", "title", "close", "hint", and "message".
        # NOTE: For simplicity, a lot of event types are just stored in the state without further processing.
        # So we incorporate them into the default case. If an event type needs special processing,
        # it should be handled explicitly in the separate branches.
        if Event == "update_session":
            self.State["update_session"].append(Data)
        elif Event == "title":
            self.State["title"] = Data.get("content")
        elif Event == "message":
            # The "message" event is the most complex, as it can contain a path, operation, and value for updating the state.
            if Data.get("p"):
                self.Path = self._NormPath(Data["p"])
            if Data.get("o"):
                self.OP = Data["o"]
            # Apply the operation to the specified path with the provided value.
            self.Apply(self.Path, self.OP, Data.get("v"))
        else:
            self.State[Event] = Data
    
'''
DeepSeekStreamParser Class
This class inherits from SSEParser and is designed to integrate SSE parsing and state decoding for DeepSeek API responses.
It will provide a much more convenient interface for handling streaming responses, extracting content and so on,
allowing users to receive incremental updates and final results.

Functions Table(Public):
- Push(Chunk):
Processes an incoming SSE event item, updating the state based on the event type and data.
- Finish():
Finish processing the stream, invoking the OnEvent callback for any remaining events and finalizing the decoder state.
- Text(Type = "response"):
Get the accumulated text content from the decoder's state, filtered by the specified type (default is "response").

Properties:
- MessageID: Get the message ID from the decoder's state.
- TokenUsage: Get the accumulated token usage from the decoder's state.
'''
class DeepSeekStreamParser(SSEParser):
    # Initialize the base SSEParser class and set up the DeepSeekStateDecoder and event callback.
    def __init__(self, OnDelta = None, OnEvent = None):
        super().__init__()
        self.Decoder = DeepSeekStateDecoder(OnDelta)
        self.OnEvent = OnEvent

    # Process an incoming SSE event item, updating the state based on the event type and data.
    def Push(self, Chunk):
        # Call the base class Push method to parse the incoming chunk into SSE events.
        Events = super().Push(Chunk)
        # For each parsed event, invoke the OnEvent callback and push it to the decoder for state management.
        for Event in Events:
            if self.OnEvent:
                self.OnEvent(Event)
            self.Decoder.Push(Event)

        return Events

    # Finish processing the stream, invoking the OnEvent callback for any remaining events and finalizing the decoder state.
    def Finish(self):
        # Call the base class Finish method to finalize the parsing of any remaining SSE events.
        Events = super().Finish()
        # For each remaining event, invoke the OnEvent callback and push it to the decoder for state management.
        for Event in Events:
            if self.OnEvent:
                self.OnEvent(Event)
            self.Decoder.Push(Event)

        return Events

    # Get the accumulated text content from the decoder's state, filtered by the specified type (default is "response").
    def Text(self, Type = "response"):
        # Retrieve the list of fragments from the decoder's state.
        Fragments = self.Decoder.State.get("message", {}).get("response", {}).get("fragments", [])
        # Concatenate the content of fragments that match the specified type, returning the combined string.
        Contents  = []
        for Fragment in Fragments:
            if isinstance(Fragment, dict) and str(Fragment.get("type", "")).upper() == Type.upper():
                Contents.append(str(Fragment.get("content", "")))
        return ''.join(Contents)

    # PROPERTY: Get the message ID from the decoder's state.
    @property
    def MessageID(self):
        # Retrieve the message ID from the decoder's state, checking both "message_id" and "id" keys.
        Message = self.Decoder.State.get("message", {})
        return Message.get("message_id") or Message.get("id")

    # PROPERTY: Get the accumulated token usage from the decoder's state.
    @property
    def TokenUsage(self):
        # Get the message and response from the decoder's state.
        Message  = self.Decoder.State.get("message", {})
        Response = Message.get("response", {})
        # Check for accumulated token usage in the response, message, or close state.
        for Object in (Response, Message, self.Decoder.State.get("close") or {}):
            for Key in ("accumulated_token_usage", "token_usage", "usage"):
                Value = None
                # If the object is a dictionary, attempt to retrieve the value for the specified key.
                if isinstance(Object, dict):
                    Value = Object.get(Key)
                # If the value is an integer or float, return it as an integer.
                if isinstance(Value, (int, float)):
                    return int(Value)
                # Otherwise, we need to check if the value is a dictionary and look for "total_tokens" or "total" keys.
                if isinstance(Value, dict):
                    for SubKey in ("total_tokens", "total"):
                        if isinstance(Value.get(SubKey), (int, float)):
                            return int(Value[SubKey])

        # If no token usage information is found, return 0.
        return 0 

# Read data from the asynchronous stream and parse it using the DeepSeekStreamParser, 
# returning the final text, thinking content, message ID, and accumulated token usage.
async def ParseStream(Body, OnDelta = None):
    # Create an instance of DeepSeekStreamParser with the provided OnDelta callback.
    Parser = DeepSeekStreamParser(OnDelta)
    # Asynchronously iterate over the incoming chunks from the Body stream.
    async for Chunk in Body:
        # Push each chunk to the parser, decoding it if it's a bytes object.
        if isinstance(Chunk, bytes):
            Parser.Push(Chunk.decode(errors = "replace"))
        else:
            Parser.Push(Chunk)
    # After all chunks have been processed, finalize the parser to handle any remaining events.
    Parser.Finish()

    # Retrieve the hint from the decoder's state to check for any errors.
    Hint = Parser.Decoder.State.get("hint")
    if isinstance(Hint, dict) and Hint.get("type") == "error":
        # If an error hint is found, raise a RuntimeError with the error content.
        raise RuntimeError(f"Deepseek API Error: {Hint.get('content', 'Unknown error')}")

    # Retrieve the response from the decoder's state, defaulting to an empty dictionary if not found.
    Response  = (Parser.Decoder.State.get("message") or {}).get("response") or {}
    MessageID = Response.get("message_id")

    # Attempt to convert the message ID to an integer, handling any exceptions that may arise.
    try:
        MessageID = int(MessageID) if MessageID is not None else None
    except (TypeError, ValueError):
        MessageID = None

    # Retrieve the accumulated token usage from the response, defaulting to -1 if not found.
    Usage = Response.get("accumulated_token_usage", -1)

    return {
        "text":                    Parser.Text("response").strip(),
        "thinking":                Parser.Text("think").strip(),
        "message_id":              MessageID,
        "accumulated_token_usage": Usage
    }
