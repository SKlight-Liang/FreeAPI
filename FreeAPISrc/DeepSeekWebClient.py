'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

The core of this module is the DeepSeekWebClient class, which 
provides an asynchronous interface to the DeepSeek web-based API.
It uses the credentials obtained after logging into the browser, 
include cookie, breaer token and user-agent, to access DeepSeek's
internal interface and maintain an authenticated session. The class 
supports implementing chat completions, file uploads, session management,
and other capabilities like handling proof-of-work challenges, post 
upload parse polling and so on. Overall, from the perspective of the 
server, this client is no different from a webpage.
'''

import json
import math
import base64
import struct
import asyncio
import hashlib
import aiohttp
import urllib.parse

from .Utils import *
from pathlib import Path

from wasmtime import Store
from wasmtime import Engine
from wasmtime import Module
from wasmtime import Instance

# Default cookie and user-agent values for the DeepSeekWebClient class,
# which will be used when no credentials are provided.
DEFAULT_COOKIE = ""
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

'''
DeepSeekWebClient Class:
This class constructs an asynchronous HTTP client for DeepSeek' internal 
API on the web page. Through maintaining an authenticated session, it can
expose various operations for users to interact with DeepSeek.

Functions Table(Public):
- Close(): Close the underlying aiohttp session.
- _Request(Method, URL, **KeywordArguments): Send an HTTP request and parse the response as JSON.
- GetBanner(): Retrieve the client settings (banner).
- CreatePowChallenge(TargetPath): Request PoW challenge for specified interface.
- _PoWChallengeSolver(Challenge): Solve the DeepSeekHash challenge using the original bundled WASM solver.
- PowChallengeSolver(Challenge): Solve the PoW challenge according to the specified algorithm.
- _BuildPowResponse(Challenge, TargetPath, Answer): Construct the PoW response payload.
- CreateChatSession(): Create a new chat session and return the session ID.
- ChatCompletions(Params): Submit a chat completion request and return an asynchronous generator for the SSE stream.
- UploadFile(Source, Filename, ModelType = "default"): Upload a file and return the file ID.
- DeleteSession(SessionID): Delete the specified chat session.
- CurrentUser(): Retrieve the information of the currently logged in user.
- GetHistory(SessionID): Retrieve message history for the specified session.
- GetSessionList(): Retrieve the list of chat sessions.
- LogOut(): Same meaning as the function name.
'''
class DeepSeekWebClient:
    # Initialize the client with standard headers and credentials, 
    # and create an aiohttp session for asynchronous requests.
    def __init__(self, Credentials, Verbose = False):
        # Standardize the credentials input to a dictionary format
        if isinstance(Credentials, str):
            try:
                # We will try to parse the credentials as a JSON string.
                Credentials = json.loads(Credentials)
            except:
                # If parsing fails, we just treat it as a cookie string 
                # and wrap it in a dictionary.
                Credentials = { "cookie": Credentials }

            # There is a situation where the credentials are still a string after
            # parsing, we also regard it as a cookie string.
            if isinstance(Credentials, str):
                Credentials = { "cookie": Credentials }

        # A switch to enable verbose logging for debugging purposes.
        self.Verbose = Verbose
        # Construct the default headers for HTTP requests, 
        # including cookie, user-agent, and other necessary fields.
        self.Headers = {
            "Cookie"    : Credentials.get("cookie"   , DEFAULT_COOKIE),
            "User-Agent": Credentials.get("userAgent", DEFAULT_USER_AGENT),
            "Accept"    : "*/*",
            "Referer"   : "https://chat.deepseek.com/",
            "Origin"    : "https://chat.deepseek.com",
            
            "x-client-platform"       : "web",
            "x-client-version"        : "2.4.0",
            "x-client-bundle-id"      : "com.deepseek.chat",
            "x-client-locale"         : "zh_CN",
            "x-client-timezone-offset": "28800",
        }
        # If a bearer token is provided in the credentials, 
        # add it to the headers for authorization.
        if Credentials.get("bearer"):
            self.Headers["Authorization"] = "Bearer " + Credentials["bearer"]
        # Create a shared aiohttp ClientSession with the constructed headers 
        # for making asynchronous HTTP requests. Every request will automatically
        # reuse this session and headers.
        self.Http = aiohttp.ClientSession(headers = self.Headers)

    # Close the underlying HTTP client session and release any associated resources.
    async def Close(self):
        await self.Http.close()

    # Send an HTTP request and parse the response as JSON. To ensure the successful
    # execution of the resolving process, the function will check the HTTP status 
    # code of the response. If the status code is over 400, an exception will be 
    # raised with the response content for debugging purposes.
    async def _Request(self, Method, URL, **KeywordArguments):
        # Initiate an asynchronous HTTP request using the specified method and URL,
        # along with any additional keyword arguments (e.g., headers, data, etc.).
        async with self.Http.request(Method, URL, **KeywordArguments) as Response:
            # Read the text response content from the server
            ResponseText = await Response.text()
            # If the HTTP status code indicates an error (400 or above), 
            # raise a RuntimeError with the status code and response content.
            if Response.status >= 400:
                raise RuntimeError(f"DeepSeek request failed: {Response.status} {ResponseText}")
            # Parse the response content as JSON.
            return json.loads(ResponseText)

    # Retrieve the client settings (banner) from the DeepSeek server
    async def GetBanner(self):
        return await self._Request(
            "GET", 
            "https://chat.deepseek.com/api/v0/client/settings?did=&scope=banner"
        )
    
    # Request a PoW challenge for a specific target path and extract its payload. 
    # We offer serveral supported response shapes to ensure compatibility with 
    # multiple versions of the challenge response. 
    async def CreatePowChallenge(self, TargetPath):
        # Call the PoW challenge creation endpoint with specified target path.
        ResponseData = await self._Request(
            "POST", 
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge", 
            json = { "target_path": TargetPath }
        )

        # Extract challenge field from the response that may appear in different nested structures.
        Challenge = (
            ((ResponseData.get("data") or {}).get("biz_data") or {}).get("challenge") or 
            (ResponseData.get("data") or {}).get("challenge") or 
            ResponseData.get("challenge")
        )
        # If we could not obtain a challenge from the response, raise an error.
        if not Challenge:
            raise RuntimeError("PoW challenge missing in response")

        return Challenge

    # Use the original bundled WASM solver to solve the DeepSeekHash challenge 
    # and decode its returned nonce. This module is obtained through package 
    # capturing and reverse engineering. For security reasons and copyright 
    # concerns, we do not provide the source code of the WASM solver. As an 
    # alternative solution, we will use wasmtime to load this module and 
    # synchronously solve PoW challenges.
    def _PoWChallengeSolver(self, Challenge):
        # Create a new wasmtime engine and store for executing the WASM module.
        EngineObject   = Engine()
        StoreObject    = Store(EngineObject)
        # Load the WASM module from the resources file.
        ModuleObject   = Module.from_file(
            EngineObject, 
            str(Path(__file__).parent / "Resources" / "DeepSeekHashV1.wasm")
        )
        # Create a new instance of the WASM module 
        InstanceObject = Instance(StoreObject, ModuleObject, [])
        
        # Extract the exported functions and memory from the WASM instance 
        # for solving the challenge, including memory allocation, stack 
        # pointer manipulation, and the main solver function.
        Exports      = InstanceObject.exports(StoreObject)
        Memory       = Exports["memory"]
        Allocate     = Exports["__wbindgen_export_0"]
        StackPointer = Exports["__wbindgen_add_to_stack_pointer"]
        Solve        = Exports["wasm_solve"]

        # Encode a string into WASM linear memory, return its pointer and length
        def Encode(Value): 
            EncodedBytes = Value.encode()
            Pointer      = Allocate(StoreObject, len(EncodedBytes), 1)
            Memory.write(StoreObject, EncodedBytes, Pointer)

            return Pointer, len(EncodedBytes)

        # Write the challenge and prefix into WASM memory
        ChallengePointer, ChallengeLength = Encode(Challenge["challenge"])
        PrefixPointer   , PrefixLength    = Encode(f"{Challenge['salt']}_{Challenge.get('expire_at')}_")
        # Allocate a return buffer in WASM memory for the solver to write its result
        ReturnPointer = StackPointer(StoreObject, -16)
        # Call the WASM solver
        Solve(
            StoreObject, 
            int(ReturnPointer), 
            int(ChallengePointer), 
            int(ChallengeLength), 
            int(PrefixPointer), 
            int(PrefixLength), 
            float(Challenge["difficulty"])
        )
        # Read the result from WASM memory
        RawAnswer = Memory.read(StoreObject, ReturnPointer, ReturnPointer + 16)
        # Restore the stack pointer after the solver call
        StackPointer(StoreObject, 16)

        # Decode the result from the WASM solver:
        # The first 4 bytes represent the status code (0 for failure, 1 for success).
        # The next 8 bytes represent the answer as a double-precision float.
        Status = struct.unpack_from("<i", RawAnswer, 0)[0]
        Answer = struct.unpack_from("<d", RawAnswer, 8)[0]
        # If the solver failed to find a solution, raise an error.
        if Status == 0:
            raise RuntimeError("DeepSeekHashV1 failed to find solution")
        # JSON.stringify will serialize an integral JavaScript Number as an integer token
        # (e.g. 42, not 42.0). To make sure our answer is consistent with the online solver,
        # we will check if the answer is an integer and return it as such.
        if not Answer.is_integer():
            raise RuntimeError(f"DeepSeekHashV1 returned a non-integral answer: {Answer!r}")

        return int(Answer)

    # Solve the PoW challenge according to the specified algorithm.
    # SHA-256: We will brute-force the nonce until we find a hash that meets the difficulty requirement.
    # DeepSeekHashV1: We will use the original WASM solver to solve the challenge.
    async def PowChallengeSolver(self, Challenge):
        # The algorithm to solve the SHA-256 challenge.
        if Challenge["algorithm"] == "sha256":
            # Calculate the difficulty in bits. If the difficulty is greater 
            # than 1000, we will take its logarithm base 2 to reduce the search space.
            Difficulty = 0
            if Challenge["difficulty"] > 1000:
                Difficulty = int(math.log2(Challenge["difficulty"]))
            else:
                Difficulty = Challenge["difficulty"]

            # Brute-force the nonce (Up to 1 million times)
            for Nonce in range(1000001):
                # Compute the SHA-256 hash of the concatenated salt, challenge, and nonce.
                HashBytes = hashlib.sha256(
                    f"{Challenge['salt']}{Challenge['challenge']}{Nonce}".encode()
                ).digest()
                # Convert the hash bytes to a binary string representation.
                Bits = "".join(f"{Byte:08b}" for Byte in HashBytes)
                # Check if the number of leading zeros in the hash meets the difficulty requirement.
                if len(Bits) - len(Bits.lstrip("0")) >= Difficulty:
                    return Nonce

            # If we exhaust the search space without finding a valid nonce, raise an error.
            raise RuntimeError("SHA-256 PoW timeout")

        # The algorithm to solve the DeepSeekHashV1 challenge.
        elif Challenge["algorithm"] == "DeepSeekHashV1":
            # Put the synchronized WASM solver into a separate thread to avoid blocking the event loop.
            return await asyncio.to_thread(self._PoWChallengeSolver, Challenge)

        raise RuntimeError("Unsupported PoW algorithm: " + Challenge["algorithm"])

    # Construct the PoW response by merging the original challenge fields 
    # with the computed answer and target path. The resulting payload will 
    # be serialized to JSON, encoded in UTF-8, and then base64-encoded for 
    # transmission. It matches the implementation of the web client:
    # JSON.stringify({ ...challenge, answer, target_path }).
    def _BuildPowResponse(self, Challenge, TargetPath, Answer):
        # Preserve the challenge fields and the order of JSON fileds
        Payload                = dict(Challenge)
        # Supplement the payload with the PoW answer and target path.
        Payload["answer"]      = int(Answer)
        Payload["target_path"] = TargetPath

        # Serialize the payload to JSON, ensuring that separators are 
        # compact and ASCII characters are preserved.
        RawPayload = json.dumps(
            Payload, separators = (",", ":"), ensure_ascii = False
        ).encode("utf-8")
        # Base64-encode the serialized JSON payload and return it as a string.
        return base64.b64encode(RawPayload).decode("ascii")

    # Create a new chat session and return the session ID
    async def CreateChatSession(self):
        # Call the chat session creation interface
        ResponseData = await self._Request(
            "POST", 
            "https://chat.deepseek.com/api/v0/chat_session/create", 
            json = {}
        )
        # Extract the session ID from the response, which may be nested in different structures.
        SessionID = ((
            (ResponseData.get("data") or {}).get("biz_data") or {}
        ).get("chat_session") or {}).get("id")
        # If the session ID is missing, raise an error.
        if not SessionID:
            raise RuntimeError("Chat session ID missing in response")

        return SessionID

    # Submit a chat completion request. To verify the identity of the client, 
    # the server will issue a PoW challenge for each request. So we should 
    # first request a challenge, solve it, and then submit the request with 
    # the PoW response. Once the request is accepted, the server will return 
    # a text/event-stream, which is a streaming response that allows the client 
    # to receive partial results in real-time. The function will return an 
    # asynchronous generator that yields the raw bytes of the SSE stream, allowing 
    # the caller to process the stream incrementally. The subsequent SSE parsing 
    # and message handling will be conducted by other components.
    async def ChatCompletions(self, Params):
        # The interface path for chat completions. 
        # This interface will give us a PoW challenge for each request, 
        # and we need to solve it before submitting the true payload.
        TargetPath = "/api/v0/chat/completion"
        # Construct the payload for the chat completion request, including session ID, 
        # model type, parent message ID, prompt, and other optional parameters. 
        Payload    = {
            "chat_session_id"  : Params["session_id"],
            "model_type"       : Params.get("model_type", "default"),
            "parent_message_id": Params.get("parent_message_id"),
            "action"           : None,
            "preempt"          : Params.get("preempt", False),
            "prompt"           : Params["message"],
            "ref_file_ids"     : Params.get("file_ids", []),
            "search_enabled"   : Params.get("search_enabled", True),
            "thinking_enabled" : Params.get("thinking_enabled", True),
        }

        # OPTIONAL: If verbose logging is enabled, print a message 
        # indicating that the chat completion request is being sent, 
        # along with the session ID and parent message ID.
        if self.Verbose:
            print("WEBCLIENT > Sending chat completion request...")
            print(f"  SessionID: {Params['session_id']}")
            print(f"  ParentMessageID: {Params.get('parent_message_id')}")

        # The challenge is one time use only and short-lived. That means we 
        # do not have much time to solve this. If the result has been explicitly 
        # rejected, we have to request a new challenge and solve it again.
        # This process requires some computation, so we will not retry too many times.
        for Attempt in range(2):
            # Request a PoW challenge from the chat completion interface.
            Challenge = await self.CreatePowChallenge(TargetPath)
            # Solve the challenge using the appropriate algorithm.
            Answer    = await self.PowChallengeSolver(Challenge)
            # Build the PoW response payload.
            Proof     = self._BuildPowResponse(Challenge, TargetPath, Answer)

            # OPTIONAL: If verbose logging is enabled, print a message 
            # indicating that the PoW challenge has been solved, 
            # along with the answer and attempt number.
            if self.Verbose:
                print("WEBCLIENT > PoW challenge solved successfully. Submitting request...")
                print(f"  Challenge: {Challenge}")
                print(f"  Answer: {Answer}")
                print(f"  Attempt: {Attempt + 1}")

            # Submit the chat completion request with the PoW response in the headers.
            Response = await self.Http.post(
                "https://chat.deepseek.com" + TargetPath, 
                json = Payload, 
                headers = { "X-Ds-Pow-Response": Proof }
            )
            # Check the response status and content type to determine how to handle the response.
            ContentType = Response.headers.get("Content-Type", "")

            # If the response status code indicates an error (400 or above), read the 
            # response text, release the connection, and raise a RuntimeError with the 
            # status code and response content.
            # NOTE: The response status is not related to whether the server accepts the PoW challenge. 
            # If everything goes well, the server will return a 200 OK status even if the PoW challenge 
            # is rejected. The rejection will be indicated by a specific error code in the response body.
            # So if the status is over 400, it means the request itself failed, or the server encountered 
            # an internal error. In that case, we will not retry anymore.
            if Response.status >= 400:
                ResponseText = await Response.text()
                Response.release()
                raise RuntimeError(f"Chat completion failed: {Response.status} {ResponseText}")

            # If the response is a SSE stream, we will return an asynchronous generator 
            # that yields the raw bytes of the stream. The generator will output the response
            # content in chunks as they arrive, until the stream is closed or the consumer 
            # stops iteration. Finally release the connection to free up resources.
            if "text/event-stream" in ContentType.lower():
                # OPTIONAL: If verbose logging is enabled, print a message indicating that the 
                # chat completion response is OK, along with the status code and content type.
                if self.Verbose:
                    print("WEBCLIENT > Chat completion response successful. Streaming response...")
                    print(f"  Status: {Response.status}")
                    print(f"  Content-Type: {ContentType}")

                # Keep the response alive until the consumer finishes or stops iteration.
                async def Generate(ResponseStream = Response):
                    try:
                        async for Chunk in ResponseStream.content.iter_any():
                            yield Chunk
                    finally:
                        ResponseStream.release()

                return Generate()

            # Otherwise, we need to read the response text and try to identify the error.
            ResponseText = await Response.text()
            Response.release()
            # Try to parse the response text as JSON to extract the error code.
            try:
                ErrorData = json.loads(ResponseText)
            except Exception:
                ErrorData = None

            # If the PoW challenge was rejected (error code 40301), 
            # we will fetch a new challenge and retry once.
            if Attempt == 0 and isinstance(ErrorData, dict) and ErrorData.get("code") == 40301:
                # OPTIONAL: If verbose logging is enabled, 
                # print a message indicating that the PoW challenge was rejected,
                if self.Verbose:
                    print("WEBCLIENT > PoW challenge rejected. Fetching a new challenge and retrying...")

                continue

            # Otherwise, we will raise a RuntimeError with the unexpected content type and response text.
            raise RuntimeError(f"Unexpected content type[{ContentType}] for chat completion response: {ResponseText}")

        # If we reach here, it means we have exhausted the retry attempts 
        # and the PoW challenge was rejected twice. Raise an error.
        raise RuntimeError("DeepSeek rejected two freshly solved PoW challenges")

    # The following method is used for uploading files to the DeepSeek server. 
    # In order to achieve a similar effect as the official API, we support 
    # various input sources, such as URLs, locah file paths, and byte-like objects.
    # The method will first load the source data, request a PoW challenge for 
    # the upload endpoint, solve it, and construct the PoW response. Then submits
    # the file as a multipart/form-data request with appropriate headers and metadata.
    # After submission, we will poll the server for processing status until the file
    # is successfully processed or an error occurs. Finally, the method returns the 
    # file ID upon successful upload and processing.
    async def UploadFile(self, Source, Filename, ModelType = "default"):
        # If the source is a URL
        if isinstance(Source, str) and Source.startswith(("http://", "https://")):
            # Download the file content from the URL using an asynchronous GET request.
            async with self.Http.get(Source) as Response:
                # If the HTTP status code indicates an error (400 or above), 
                # raise a RuntimeError with the status code and response text.
                if Response.status >= 400:
                    raise RuntimeError(f"Failed to fetch upload source: {Response.status} {await Response.text()}")
                # Read the response content as bytes for upload.
                Data = await Response.read()
        
        # If the source is just a regular string
        elif isinstance(Source, str):
            # Attempt to decode the string as a data URL. 
            Decoded = DecodeDataURL(Source)
            if Decoded is not None:
                Data = Decoded
            else:
                # Otherwise, treat the string as a local file path and read its bytes.
                Data = Path(Source).read_bytes()

        # In other cases, the source will be considered as byte sequence-like object
        else:
            Data = bytes(Source)

        # The interface path for file upload. 
        # This is similar to the chat completion interface.
        TargetPath = "/api/v0/file/upload_file"
        # Request a PoW challenge and build the PoW response payload.
        Challenge = await self.CreatePowChallenge(TargetPath)
        Answer    = await self.PowChallengeSolver(Challenge)
        Proof     = self._BuildPowResponse(Challenge, TargetPath, Answer)

        # Construct a multipart/form-data request with the file content, filename, and content type.
        Form = aiohttp.FormData()
        Form.add_field(
            "file", Data, 
            filename     = Filename, 
            content_type = DetectMIMEType(Data, Filename)
        )
        # Construct the upload headers and remove the content-type header, 
        # which will be automatically set by aiohttp.
        UploadHeaders = {
            Key: Value for Key, Value in self.Headers.items()
                if Key.lower() != "content-type"
        }
        # Add the PoW response, file size, and model type to the upload headers.
        UploadHeaders.update({
            "X-Ds-Pow-Response": Proof,
            "X-File-Size"      : str(len(Data)),
            "X-Model-Type"     : ModelType or "default",
        })

        # Submit the file upload request with the constructed form data and headers.
        # The server will respond with a JSON object containing the file ID and processing status.
        async with self.Http.post(
            "https://chat.deepseek.com" + TargetPath, 
            data = Form, headers = UploadHeaders
        ) as Response:
            # Read the response text
            ResponseText = await Response.text()
            # Check the response status code. If it indicates an error (400 or above), 
            # raise a RuntimeError with the status code and response text.
            if Response.status >= 400:
                raise RuntimeError(f"File upload failed: {Response.status} {ResponseText}")

            # Try to parse the response text as JSON.
            try:
                ResponseData = json.loads(ResponseText)
            except Exception as Error:
                raise RuntimeError(
                    f'File upload returned non-JSON response: status={Response.status}, ' 
                    f'content-type={Response.headers.get("Content-Type", "")}, '
                    f'body={ResponseText[:500]!r}'
                ) from Error

        # Extract the business data from the response, 
        # which contains the file ID and processing status.
        BusinessData = ((ResponseData.get("data") or {}).get("biz_data") or {})
        # Retrieve the file ID from the business data.
        FileID = BusinessData.get("id") or BusinessData.get("file_id")
        # If the file ID is missing, raise an error.
        if not FileID:
            raise RuntimeError(f"File ID missing in upload response for {Filename}: {ResponseData}")

        # Check the processing status of the uploaded file.
        Status = BusinessData.get("status")
        if Status == "SUCCESS":
            # The file has been successfully uploaded and processed. Return the file ID.
            return FileID
        if Status == "FAILED":
            # The file upload failed. Raise an error with the filename and file ID.
            raise RuntimeError(f"File upload failed for {Filename} with fileId {FileID}")
        if Status == "CONTENT_EMPTY":
            # The file upload failed due to empty content. Raise an error with the filename and file ID.
            raise RuntimeError(f"File upload failed due to empty content for {Filename} with fileId {FileID}")

        # If the file is still being processed, we will poll the server for its status.
        # We will check up to 30 times with a two-second delay between each poll. 
        # Therefore, the maximum wait time for processing is 60 seconds. Even if the 
        # processing is not completed within this time frame, we will still end the polling.
        for _ in range(30):
            # Wait for 2 seconds before polling the server again.
            await asyncio.sleep(2)
            # Construct the URL for fetching the file status using the file ID.
            URL = "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + urllib.parse.quote(str(FileID))
            # Send a GET request to fetch the file status.
            async with self.Http.get(URL) as Response:
                # If the HTTP status code indicates an error (400 or above), continue to the next poll.
                if Response.status >= 400:
                    continue
                # Read the response text and try to parse it as JSON.
                ResponseText = await Response.text()
                try:
                    PollData = json.loads(ResponseText)
                except Exception:
                    continue
            # Extract the list of files from the polling response.
            Files = ((((PollData.get("data") or {}).get("biz_data") or {}).get("files")) or [])
            # Get the first file's data if available, otherwise use an empty dictionary.
            FileData = Files[0] if Files else {}
            # Check the processing status of the file.
            if FileData.get("status") == "SUCCESS":
                return FileID
            if FileData.get("status") == "FAILED":
                raise RuntimeError(f"File parsing failed for {Filename} with fileId {FileID}")
            if FileData.get("status") == "CONTENT_EMPTY":
                raise RuntimeError(f"File parsing failed due to empty content for {Filename} with fileId {FileID}")

        # If we reach here, it means the file upload polling has timed out.
        # OPTIONAL: If verbose logging is enabled, print a message indicating that 
        # the polling has timed out for the given file ID.
        if self.Verbose:
            print(f"WEBCLIENT > File upload polling timed out for {Filename} with fileId {FileID}")
    
        return FileID

    # Delete the specified session
    async def DeleteSession(self, SessionID):
        # Send a POST request to the chat session deletion endpoint with the session ID.
        return await self._Request(
            "POST", 
            "https://chat.deepseek.com/api/v0/chat_session/delete", 
            json = { "chat_session_id": SessionID }
        )

    # Retrieve the authenticated account profile of the current user.
    async def CurrentUser(self):
        # Send a GET request to the current user endpoint to fetch the profile information.
        return await self._Request(
            "GET", 
            "https://chat.deepseek.com/api/v0/users/current"
        )

    # Retrieve message history for the specified session.
    async def GetHistory(self, SessionID):
        # Send a GET request to the chat history endpoint 
        # with the session ID as a query parameter.
        return await self._Request(
            "GET", 
            f"https://chat.deepseek.com/api/v0/chat/history_messages?chat_session_id={urllib.parse.quote(SessionID)}"
        )

    # Retrieve the session list
    async def GetSessionList(self):
        # Send a GET request to the chat session list endpoint to fetch all chat sessions.
        # NOTE: The number of sessions obtained will be limited to 100. This means that 
        # only the last 100 messages could be captured by this interface.
        return await self._Request(
            "GET",
            "https://chat.deepseek.com/api/v0/chat_session/fetch_page?count=100"
        )

    # Logout the current user and invalidate the session.
    async def LogOut(self):
        # Send a POST request to the logout endpoint to log out the current user.
        return await self._Request(
            "POST",
            "https://chat.deepseek.com/api/v0/users/logout",
            json = { }
        )
