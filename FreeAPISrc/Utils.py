'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module provides MIME type detection and data URL decoding utilities.
It offers underlying functions for file and image transfer from the client to the LLM server.
At the same time, it can also be used for parsing and constructing the "Response ID", 
which is a unique identifier for each response in the WebAI system.

Here we have to mention waht the "Response ID" is like.
"Response ID" is a string consiting of 2 parts: a "Session ID" and a optional "Message ID",
in the format of "SessionID|MessageID" or just "SessionID".


Functions Table:
- DetectMIMEType(DataorPath, Filename = None): 
Detects the MIME type based on the provided data/path and optional filename.
- DecodeDataURL(Value):
Decodes a data URL into its binary content.
- ConstructResponseID(SessionID, MessageID = None):
Constructs a "Response ID" based on the given Session ID and optional Message ID.
- ParseResponseID(Value):
Parses a "Response ID" into its Session ID and Message ID components.
- EstimateTokenUsage(TotalTokens, Input, Output):
Estimate input/output token allocation based on total tokens and the input/output ratio.
'''

import base64
import pathlib
import mimetypes

# Detect the MIME type based on the provided data/path and optional filename
def DetectMIMEType(DataorPath, Filename = None):
    # Determine the parameter used for guessing MIME types
    Name = ""
    # We will prioritize using explicitly passed Filename 
    if Filename is not None:
        Name = Filename
    # If not available, try to use the DataorPath if it is a string or pathlib.Path object
    else:
        if isinstance(DataorPath, (str, pathlib.Path)):
            Name = str(DataorPath)

    # Guess the MIME type based on the Name
    MIME, _ = mimetypes.guess_type(Name)
    if MIME is None or MIME == "":
        # If the MIME type could not be determined, default to the generic binary stream type
        return "application/octet-stream"
    return MIME

# Decode a data URL into its binary content. If the input is not a valid data URL, return None
def DecodeDataURL(Value):
    # Determine if the input is a data URL
    if Value.startswith("data:"):
        # Split the data URL into its header and data components by the first comma
        # Here we guarantee that commas in the data will not be further separated
        Head, Data = Value.split(",",1)
        # Check if the header declares the data as base64 encoded.
        if ";base64" in Head:
            return base64.b64decode(Data)
        else:
            return Data.encode()

    return None

# Construct a "Response ID" based on the given Session ID and optional Message ID. 
def ConstructResponseID(SessionID, MessageID = None):
    # Note that the Session ID is mandatory, while the Message ID is optional.
    if SessionID is None:
        return ""

    if MessageID is None:
        # If the Message ID is not provided, return only the Session ID
        return SessionID.strip()
    else:
        # If the Message ID is provided, return the combined "SessionID|MessageID" format
        return f"{SessionID.strip()}|{MessageID}"

# Parse a "Response ID" into its Session ID and Message ID components.
def ParseResponseID(Value):
    if Value is None:
        return None, None
    
    # Convert the passed in value into a string and strip any leading/trailing whitespace
    Raw = Value.strip()
    # If the value does not contain a pipe character, it is assumed to be just the Session ID
    if "|" not in Raw:
        # If the length of the Session ID is less than 10 characters, it is considered invalid
        if len(Raw) < 10:
            return None, None
        
        return Raw, None

    # Split the value into Session ID and Message ID components at the first occurrence of the pipe character
    SessionID, MessageID = Raw.split("|", 1)
    SessionID = SessionID.rstrip()
    # If the length of the Session ID is less than 10 characters, it is considered invalid
    if len(SessionID) < 10:
        return None, None

    # Attempt to convert the Message ID to an integer, if it is provided
    if MessageID.strip():
        try:
            MessageID = int(MessageID.lstrip())
            # If the Message ID is negative, it is considered invalid
            if MessageID < 0:
                MessageID = None
        except ValueError:
            MessageID = None
    else:
        MessageID = None

    return SessionID, MessageID

# Estimate the proportion of input/output tokens 
# based on the total number of tokens and the ratio of input/output characters.
def EstimateTokenUsage(TotalTokens, Input, Output):
    # Roughly convert the number of input and output characters to the number of tokens by dividing them by 4
    InputWeight  = len(Input) >> 2
    OutputWeight = len(Output) >> 2
    TotalWeight  = InputWeight + OutputWeight

    # Calculate the estimated number of input tokens
    if TotalWeight > 0:
        InputTokens = (TotalTokens * InputWeight) // TotalWeight
    else:
        InputTokens = 0

    return {
        'input_tokens': InputTokens,
        'output_tokens': TotalTokens - InputTokens,
        'total_tokens': TotalTokens
    }
