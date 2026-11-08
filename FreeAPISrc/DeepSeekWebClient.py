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
            str(Path(__file__).parent / "Resources" / "DeepseekHashV1.wasm")
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
