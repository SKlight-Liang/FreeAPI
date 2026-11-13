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

import re
import sys
import copy
import json
import codecs

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
- _SyncOutputs():
Synchronizes the outputs by checking for new content in the fragments and invoking the OnDelta callback.
- _CaptureReferences(Before, After, Batch):
Captures the references before and after a batch operation to preserve bindings between references and citations.
- Apply(Path, Operation, Value):
Applies the specified operation (SET, APPEND, or BATCH) to the value at the given path.
- Push(Item):
Processes an incoming SSE event item, updating the state based on the event type and data.
'''
class DeepSeekStateDecoder:
    # Initialize the decoder, and set the initial state, default path, 
    # default operation, and incremental callback.
    # FEATURE: We add OnTool callback to handle tool events. The function
    # will be called when the tool states are updated. And the user can use
    # this callback to learn more about how the model is thinking.
    def __init__(self, OnDelta = None, OnTool = None):
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
        self.OnTool  = OnTool

        # Save the emitted content of each text fragment to avoid duplicate output.
        self.EmittedText       = {}
        # Save the latest tool fragment status the parser has observed. 
        # This is used to detect changes in the tool state and trigger the OnTool callback.
        self.ToolSnapshots     = {}
        # Preserve citation bindings across references SET operations.
        self.ReferenceRegistry = {}

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

    # FEATURE: We have redesigned the incremental callback process and 
    # added a judgment function for incremental text. If a certain text 
    # has already been output before, it will not trigger a new output.
    def _SyncOutputs(self):
        # Get the list of fragments from the current state.
        Fragments = (self.State.get("message") or {}).get("response", {}).get("fragments", [])
        # Just a safety check to ensure Fragments is a list before proceeding.
        if not isinstance(Fragments, list):
            return
        
        # Iterate through the fragments
        for Index, Fragment in enumerate(Fragments):
            # Just a safety check to ensure Fragment is a dictionary.
            # In most cases, it should be a dictionary with "type" and "content" keys.
            if not isinstance(Fragment, dict):
                continue

            # Get the type of the fragment to determine how to handle it.
            Kind     = str(Fragment.get("type", "")).upper()
            # ID will be used to confirm which message this fragment belongs to.
            # That is because in the streaming process, a message may be split into
            # multiple fragments, and we have to handle them properly. 
            Identity = Fragment.get("id")
            # Construct the unique key for the message based on its type and ID.
            if Identity is not None:
                Key = (Kind, Identity)
            else:
                Key = (Kind, Index)

            # For text fragments, we will check if the content has already been emitted before.
            if Kind in ("RESPONSE", "THINK"):
                # Extract the content of the current fragment.
                Current = Fragment.get("content", "")
                # A simple safety check to ensure Content is a string
                if not isinstance(Current, str):
                    continue
                # Find the message to which this fragment belongs. 
                # This is important for handling multiple messages in a stream.
                Previous = self.EmittedText.get(Key, "")

                if Current.startswith(Previous):
                    # Calculate the new content that has not been emitted yet.
                    Delta = Current[len(Previous) : ]
                    # If there is new content, invoke the OnDelta callback with the type and new content.
                    if Delta and self.OnDelta:
                        self.OnDelta(Kind, Delta)
                    # Update the emitted text for this message to include the new content.
                    self.EmittedText[Key] = Current
                elif Current != Previous:
                    # If the current content does not start with the previous content, 
                    # it means the message has changed or been reset. We will emit the new content.
                    # WARNING: This is a vary aggressive approach, and it may lead to duplicate 
                    # output in some cases. But we want to see all the actions of the model.
                    if self.OnDelta:
                        self.OnDelta(Kind, Current)
                    # Update the emitted text for this message to the new content.
                    self.EmittedText[Key] = Current
            # For tool fragments, we will check if the tool state has changed since the last observation.
            elif Kind.startswith("TOOL_"):
                # Get the previous snapshot of the tool state for this message.
                Previous = self.ToolSnapshots.get(Key)
                # If the current tool state is different from the previous snapshot,
                # update the snapshot and invoke the OnTool callback with the new tool state.
                if Fragment != Previous:
                    # Create a deep copy of the current tool state.
                    Snapshot                = copy.deepcopy(Fragment)
                    self.ToolSnapshots[Key] = Snapshot

                    if self.OnTool:
                        self.OnTool(copy.deepcopy(Snapshot))

    # FEEATURE: We notice that the references in the response do not have a one-to-one
    # correspondence with the citations in the tool snapshots. So we need to maintain
    # a registry to preserve the bindings between references and citations across SET 
    # operations. We will bind citation occurrences to tool IDs before a later SET
    # replaces the references.
    def _CaptureReferences(self, Before, After, Batch):
        # A simple safety check to ensure Before and After are both lists before proceeding.
        if not isinstance(Before, list) or not isinstance(After, list):
            return

        # Locate reference writes within the batch. There paths are relative to the 
        # root of the batch, which is normally "response/fragments/-1".
        Writes = []
        for Item in Batch:
            if (
                isinstance(Item, dict) and 
                str(Item.get("p", "")).split('/')[-1] == "references" and
                isinstance(Item.get("v"), list)
            ):
                Writes.append(Item["v"])
        # If there are no reference writes, we will not do anything.
        if not Writes:
            return
        
        # Construct a mapping of fragment IDs to their corresponding fragments from the "Before" state.
        BeforeMap = {Fragment.get("id"): Fragment for Fragment in Before if isinstance(Fragment, dict)}
        # Iterate through the "After" fragments to find those that have been updated with new references.
        for Fragment in After:
            # In most cases, only the response contains the reference that we need to parse.
            # So we will check if the parent of the references is a response. 
            if not isinstance(Fragment, dict) or Fragment.get("type") != "RESPONSE":
                continue

            # Get the fragment ID to locate the corresponding fragment in the "Before" state.
            FragmentID = Fragment.get("id")
            if FragmentID is None:
                continue

            # Get the old and new content of the fragment to determine if there are any changes.
            OldContent = (BeforeMap.get(FragmentID) or {}).get("content") or ""
            NewContent = Fragment.get("content") or ""
            # Find all reference markers in the old and new content using regular expressions.
            OldMarkers = list(re.finditer(r"\[reference:(\d+)\]", OldContent))
            NewMarkers = list(re.finditer(r"\[reference:(\d+)\]", NewContent))
            # If the new content does not start with the old content or if the number of new markers 
            # is less than or equal to the number of old markers, we will skip this fragment.
            if not NewContent.startswith(OldContent) or len(NewMarkers) <= len(OldMarkers):
                continue

            # In our observation, the citation batch will append markers and write
            # the corresponding references in the same transaction. We will try to 
            # capture the references and bind them to the corresponding markers.
            AddedMarkers = len(NewMarkers) - len(OldMarkers)
            Values       = Writes[-1]
            # If the number of values in the last write is less than the number of 
            # added markers, we will skip this fragment because we cannot bind them correctly.
            if len(Values) < AddedMarkers:
                continue

            # We will bind the last AddedMarkers values to the corresponding markers in the fragment.
            Registry = self.ReferenceRegistry.setdefault(FragmentID, {})
            for Offset, Reference in enumerate(Values[-AddedMarkers:]):
                Registry[len(OldMarkers) + Offset] = copy.deepcopy(Reference)

    # Apply the specified operation (SET, APPEND, or BATCH) to the value at the given path.
    def Apply(self, Path, Operation, Value):
        # Normalize the path to ensure it starts with "message" and is in the correct format.
        NormPath = self._NormPath(Path)
        # Split the normalized path into tokens for traversal and manipulation.
        Tokens   = [x for x in NormPath.split('/') if x]

        # Handle batch operations by recursively applying each operation in the list.
        if Operation == "BATCH" and isinstance(Value, list):
            # FEATURE: We will capture the references before the batch operation is applied,
            # so that we can preserve the bindings between references and citations across SET operations.
            Before = copy.deepcopy((self.State.get("message") or {}).get("response", {}).get("fragments", []))

            for Item in Value:
                if isinstance(Item, dict):
                    SubPath = f"{NormPath}/{Item['p']}" if Item.get('p') else NormPath
                    SubOp   = Item.get('o', "SET")
                    SubVal  = Item.get('v')
                    self.Apply(SubPath, SubOp, SubVal)

            # FEATURE: We will capture the references after the batch operation is applied.
            After = (self.State.get("message") or {}).get("response", {}).get("fragments", [])
            # We will capture the references after the batch operation is applied.
            self._CaptureReferences(Before, After, Value)
            
            return

        # For non-batch operations, apply the delta and set the value accordingly.
        # BUG: We have found that the SSE events did not necessarily represent new content.
        # This might lead to duplicate output in some cases. So here we will exchange the 
        # order of two operations: Update the status first, then calculate the content 
        # that needs to be output.
        self.SetValue(Tokens, Value, Operation == "APPEND")
        self._SyncOutputs()

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
- ResolveReferences():
Resolves the captured references to their actual values.

Properties:
- MessageID: Get the message ID from the decoder's state.
- TokenUsage: Get the accumulated token usage from the decoder's state.
'''
class DeepSeekStreamParser(SSEParser):
    # Initialize the base SSEParser class and set up the DeepSeekStateDecoder and event callback.
    # FEATURE: Extend the OnTool callback interface to handle tool events.
    def __init__(self, OnDelta = None, OnEvent = None, OnTool = None):
        super().__init__()
        self.Decoder = DeepSeekStateDecoder(OnDelta, OnTool)
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

    # FEATURE: We have built a registry to preserve the bindings between references and 
    # citations across SET operations. Here we will resolve the references in the response 
    # to their corresponding citations in the tool snapshots.
    def ResolveReferences(self):
        # Retrieve the list of fragments from the decoder's state.
        Fragments = self.Decoder.State.get("message", {}).get("response", {}).get("fragments", [])
        # Construct a mapping of tool fragments by their IDs for easy lookup.
        ToolMap   = {
            Fragment.get("id"): Fragment for Fragment in Fragments
            if isinstance(Fragment, dict) and Fragment.get("id") is not None 
            and str(Fragment.get("type", "")).upper().startswith("TOOL_")
        }

        # FEATURE: We will construct a map between the reference markers and their corresponding 
        # citations. In this way, once a reference cannot be parsed correctly, we can still try 
        # to restore it from the list.
        URLMetaData = {}

        # Iterate through the fragments to find tool fragments and extract their URLs, 
        # titles, and site names for reference resolution.
        for Fragment in Fragments:
            # Only the tool fragments contain the information we need to resolve references.
            if not isinstance(Fragment, dict):
                continue
            # Get the type of the fragment to determine if it is a tool fragment.
            Kind = Fragment.get("type")

            if Kind == "TOOL_SEARCH":
                # Extract the results from the tool search fragment
                Items = Fragment.get("results") or []
            elif Kind == "TOOL_OPEN":
                # Extract the result from the tool open fragment
                Result = Fragment.get("result")
                Items  = [Result] if isinstance(Result, dict) else []
            else:
                continue

            # Iterate through the items to extract the URL, title, and site name for each citation.
            for Item in Items:
                if not isinstance(Item, dict):
                    continue
                # Get the URL from the item.
                URL = Item.get("url")
                if not isinstance(URL, str) or not URL:
                    continue

                Metadata = URLMetaData.setdefault(URL, {})
                # If the title is not already set in the metadata, set it from the item.
                for Field in ("title", "site_name", "snippet"):
                    if Item.get(Field) and not Metadata.get(Field):
                        Metadata[Field] = Item[Field]

        ResolvedReferences = []
        # Iterate through the fragments to find references and 
        # resolve them to their corresponding citations.
        for Fragment in Fragments:
            # Only the response fragments contain references that we need to resolve.
            if not isinstance(Fragment, dict) or Fragment.get("type") != "RESPONSE":
                continue

            # Get the fragment ID and content for the current response fragment.
            FragmentID = Fragment.get("id")
            Content    = Fragment.get("content") or ""
            # Find the reference markers in the content using a regular expression.
            Markers    = list(re.finditer(r"\[reference:(\d+)\]", Content))
            # Get the registry for the current fragment we have built before 
            # to search for the corresponding citations to bind the references to.
            Saved      = self.Decoder.ReferenceRegistry.get(FragmentID, {})

            FinalReferences = Fragment.get("references") or []
            # Iterate through the markers to resolve each reference to its corresponding citation.
            for Ordinal, Match in enumerate(Markers):
                # Get the reference from the saved registry.
                Reference = Saved.get(Ordinal)
                # If the reference is not found in the saved registry, 
                # try to get it from the final references list.
                if Reference is None and len(FinalReferences) == len(Markers):
                    Reference = FinalReferences[Ordinal]
                # Normalize the reference to a dictionary if it is not already one.
                Reference = Reference if isinstance(Reference, dict) else {}

                # Extract the tool ID and type from the reference for further processing.
                ToolID   = Reference.get("id")
                ToolType = Reference.get("type")
                # Attempt to find the corresponding tool fragment by its ID.
                Tool     = ToolMap.get(ToolID)

                # FEATURE: Initialize the URLs, Title, and SiteName variables to None. 
                # These will be used to store the resolved reference information.
                URLs     = []
                Title    = None
                SiteName = None

                if ToolType == "TOOL_OPEN":
                    # If a corresponding tool fragment is found, get the result of the tool execution.
                    Result = Tool.get("result") if isinstance(Tool, dict) else None
                    # Normalize the result to a dictionary
                    Result = Result if isinstance(Result, dict) else {}

                    # If the result contains a URL, we will use it as the resolved reference.
                    if Result.get("url"):
                        URLs = [Result["url"]]

                    Title    = Result.get("title")
                    SiteName = Result.get("site_name")

                elif ToolType == "TOOL_SEARCH":
                    # Extract the URLs from the results
                    URLs = [
                        URL for URL in Reference.get("urls", [])
                        if isinstance(URL, str) and URL
                    ]

                    # If a corresponding tool fragment is found, get the result of the tool execution.
                    if isinstance(Tool, dict):
                        Results = Tool.get("results") or []
                        # Iterate through the results to find a matching URL and extract the title and site name.
                        for Item in Results:
                            if isinstance(Item, dict) and Item.get("url") in URLs:
                                Title    = Item.get("title")
                                SiteName = Item.get("site_name")
                                break

                # FEATURE: If the reference cannot be correctly resolved, 
                # we will try to restore it from the URL metadata we have built before.
                if URLs:
                    Metadata = URLMetaData.get(URLs[0], {})
                    Title    = Title    or Metadata.get("title")
                    SiteName = SiteName or Metadata.get("site_name")

                # Bind the reference to the result of the tool execution,
                # and add it to the list of resolved references.
                ResolvedReferences.append({
                    "index"               : int(Match.group(1)),
                    "response_fragment_id": FragmentID,
                    "ordinal"             : Ordinal,
                    "tool_fragment_id"    : ToolID,
                    "type"                : ToolType,
                    "url"                 : URLs[0] if URLs else None,
                    "urls"                : URLs,
                    "title"               : Title,
                    "site_name"           : SiteName,
                })

        return ResolvedReferences

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

# DEBUG: The system will output thinking contents multiple times, which is not ideal.
# This function is designed to output the raw SSE stream for debugging purposes.
def DebugEventStream(Event):
    if Event.get("event") == "message":
        print(
            "RAW SSE > ",
            json.dumps(Event.get("data"), ensure_ascii = False, indent = 2),
            file = sys.stderr
        )

# Read data from the asynchronous stream and parse it using the DeepSeekStreamParser, 
# returning the final text, thinking content, message ID, and accumulated token usage.
# FEATURE: We have added a new OnTool callback to handle tool events.
# FEATURE: We have added a new OnEvent callback to log the raw SSE events for debugging purposes.
async def ParseStream(Body, OnDelta = None, OnTool = None, OnEvent = None):
    # Create an instance of DeepSeekStreamParser with the provided OnDelta callback.
    Parser = DeepSeekStreamParser(OnDelta = OnDelta, OnTool = OnTool, OnEvent = OnEvent)
    # FEATURE: Add a UTF-8 incremental decoder to handle any potential decoding 
    # issues with the incoming byte stream.
    UTF8Decoder = codecs.getincrementaldecoder("utf-8")("replace")
    # Asynchronously iterate over the incoming chunks from the Body stream.
    async for Chunk in Body:
        # Push each chunk to the parser, decoding it if it's a bytes object.
        if isinstance(Chunk, bytes):
            Parser.Push(UTF8Decoder.decode(Chunk, final = False))
        else:
            Parser.Push(Chunk)
    # After all chunks have been processed, finalize the parser to handle any remaining events.
    Parser.Push(UTF8Decoder.decode(b"", final = True))
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

    ToolSearch = [ 
        copy.deepcopy(Fragment) for Fragment in Response.get("fragments", [])
        if isinstance(Fragment, dict) and Fragment.get("type") == "TOOL_SEARCH"
    ]
    ToolOpen   = [
        copy.deepcopy(Fragment) for Fragment in Response.get("fragments", [])
        if isinstance(Fragment, dict) and Fragment.get("type") == "TOOL_OPEN"
    ]

    return {
        "text"                   : Parser.Text("response").strip(),
        "thinking"               : Parser.Text("think").strip(),
        "fragments"              : copy.deepcopy(Response.get("fragments", [])),
        "searches"               : ToolSearch,
        "opens"                  : ToolOpen,
        "references"             : Parser.ResolveReferences(),
        "reference_registry"     : copy.deepcopy(Parser.Decoder.ReferenceRegistry),
        "message_id"             : MessageID,
        "accumulated_token_usage": Usage
    }
