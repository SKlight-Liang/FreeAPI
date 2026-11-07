'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module defines a SSE parser to parse Server-Sent Events (SSE) streams.
The SSEParser class could progressively receive data chunks and parse out complete SSE events.
It supports both the "event" and "data" fields of SSE messages, and can handle JSON-formatted data in the "data" field. 
'''

import json

'''
SSEParser Class
Analyze the data stream in Server Sent Events (SSE) format.

Functions Table(Public):
- Push(Chunk):
Accepts a new chunk of data, appends it to the internal buffer, and parses all complete events from the buffer. 
- Finish():
Called at the end of the data stream to process any remaining data in the buffer as the last event.
'''
class SSEParser:
    # Initialize SSEParser instance: Set the internal buffer to an empty string
    def __init__(self): 
        self.Buffer = ""

    # Process newly arrived data chunks, add them to the buffer, parse all complete events
    # And finally return a list of events
    def Push(self, Chunk):
        self.Buffer += Chunk.replace("\r\n","\n")

        # Initialize the output list to store the events parsed this time.
        OutputList = []

        while "\n\n" in self.Buffer:
            # Retrieve the current complete event block
            Block, self.Buffer = self.Buffer.split("\n\n", 1)
            # Initialize the event type to the default "message" and create a list for collecting data rows
            Event = "message"
            Data  = []

            for Line in Block.splitlines():
                if Line.startswith("event:"): 
                    # Extract the event type and remove the leading and trailing blanks
                    Event = Line[6:].strip()
                elif Line.startswith("data:"): 
                    # Extract the data portion and append it to the data list
                    Data.append(Line[5:].lstrip())
            
            if Data:
                Raw = "\n".join(Data)
                try:
                    # Attempt to parse the data into a JSON object
                    JSONObject = json.loads(Raw)
                except Exception: 
                    # If parsing fails, use the raw data
                    JSONObject = Raw

                OutputList.append({
                    "event": Event,
                    "data": JSONObject
                })
        
        return OutputList

    # Called at the end of the data stream. If there is still remaining data in the buffer,
    # it will be processed as the last event and return a list of events.
    def Finish(self):
        if self.Buffer.strip(): 
            return self.Push("\n\n")
        return []
