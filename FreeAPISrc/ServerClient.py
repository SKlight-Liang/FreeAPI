'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module defines a client abstraction layer for server-side use,
encapsulating two types of backends that interact with DeepSeek:
- API-backed adapter that communicates with DeepSeek via HTTP requests.
- Browser-backed adapter that interacts with DeepSeek through a browser 
automation framework.

Both adapters expose consistent interfaces to the upper layers, allowing 
seamless switching between them without altering the underlying client 
behavior. The interfaces include methods for creating sessions, deleting 
sessions, chat completions and uploading files.

Functions Table:
- _SourcetoBytesStream(Source):
Read the upload source and convert it into a byte stream uniformly.
- CreateServerClient(CredentialPath, BrowserMode = False, UserDataDir = None):
Create corresponding server client instance based on the specified mode.
'''

import aiohttp

from pathlib import Path

from .Utils import *
from .Browser import *
from .DeepSeekAuth import *
from .DeepSeekWebClient import *
from .DeepSeekBrowserClient import *

# Convert various formats of upload sources into a byte stream uniformly.
# The format can be a local file path, a URL/data URL, or just a raw byte stream.
async def _SourcetoBytesStream(Source):
    # If the source is already a byte stream, return it directly.
    # Here we assume that the source is a byte stream as long as it is not a string. 
    # This is a design choice to simplify the interface.
    if not isinstance(Source, str):
        return bytes(Source)
    
    # Try to decode the source as a data URL. If successful, return the decoded byte stream.
    DecodedData = DecodeDataURL(Source)
    if DecodedData is not None:
        return DecodedData
    
    # If the source is a URL (starting with http:// or https://), 
    # fetch the content from the URL.
    if Source.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as Session:
            async with Session.get(Source) as Response:
                # If the HTTP response status code indicates an error (400 or above), 
                # raise a RuntimeError with the status code and response text.
                if Response.status >= 400:
                    raise RuntimeError(f"Failed to fetch upload source: {Response.status} {await Response.text()}")
                # Otherwise, return the content of the response as a byte stream.
                return await Response.read()

    # In other cases, treat the source as a local file path and read its content as a byte stream.
    return Path(Source).read_bytes()

'''
ApiServerClient Class
This class adapts the request-based DeepSeek web client to the server interface, 
providing methods for creating sessions, deleting sessions, chat completions, and 
uploading files. It uses the underlying DeepSeekWebClient to perform these operations, 
while managing session IDs and handling upload sources uniformly.

Functions Table(Public):
- Create(Path): Load and create an API client instance from the given credentials path.
- ChatCompletion(Params): Initiate a chat completion request, reusing or creating a session ID as needed.
- UploadFile(Source, Filename, ModelType): Read the source byte stream and upload it to the API.
- DeleteSession(SessionID): Delete a specified session by its session ID.
- Close(): Close the underlying web client and release resources.
'''
class ApiServerClient:
    # Define the mode identifier for this client as "API", 
    # indicating that it uses the API-backed adapter.
    Mode = "API"

    # Initialize the ApiServerClient with an instance of DeepSeekWebClient.
    def __init__(self, Client):
        self.Client = Client

    # Create an instance of ApiServerClient from credentials stored at the given path.
    @classmethod
    async def Create(cls, Path):
        return cls(DeepSeekWebClient(LoadCredentials(Path)))

    # Initiate a chat completion request. Reuse an existing session ID if provided,
    # or create a new session ID before forwarding the request to the underlying client.
    # The method returns a tuple containing the session ID and the result of the chat 
    # completion request.
    async def ChatCompletion(self, Params):
        SessionID = Params.get("session_id") or await self.Client.CreateChatSession()
        return SessionID, await self.Client.ChatCompletions({
            **Params, "session_id": SessionID
        })

    # Upload a file to the DeepSeek API. The method decodes the upload source (which can 
    # be a local file path, a URL, or a data URL) into a byte stream and forwards it to 
    # the underlying API client along with the specified filename and model type. 
    # The method returns the result of the upload operation including the file ID and 
    # any other relevant information.
    async def UploadFile(self, Source, Filename, ModelType = "default"):
        return await self.Client.UploadFile(
            await _SourcetoBytesStream(Source), Filename, ModelType
        )

    # Delete a specified session by its session ID using the underlying API client. This method 
    # allows for cleanup of resources associated with a session when it is no longer needed.
    async def DeleteSession(self, SessionID):
        await self.Client.DeleteSession(SessionID)
    
    # Close the ApiServerClient instance and release any resources held by the underlying API client.
    async def Close(self):
        await self.Client.Close()

'''
BrowserServerClient Class
This class defines a server-side client that interacts with DeepSeek through a browser 
automation framework, specifically using Playwright to control a Chromium browser instance. 
It manages the lifecycle of the browser and the underlying API client.

Functions Table(Public):
- Create(UserDataDir): Start browser debugging, connect and create a BrowserServerClient instance.
- ChatCompletions(Params): Initiate a chat completion request through the browser client.
- UploadFile(Source, Filename, ModelType): Read the source byte stream and upload it.
- DeleteSession(SessionID): Delete a specified session by its session ID.
- Close(): Close the browser and Playwright instances, and terminate the browser process.
'''
class BrowserServerClient:
    # Define the mode identifier as "browser", making it clear that 
    # this client uses a browser-backed adapter for interacting with DeepSeek.
    Mode = "Browser"

    # Store the Playwright instance, the browser instance, the process handle, and the
    # underlying DeepSeekBrowserClient instance for managing interactions with DeepSeek.
    def __init__(self, Playwright, Browser, Process, Client):
        self.Playwright = Playwright
        self.Browser    = Browser
        self.Process    = Process
        self.Client     = Client

    # Launch a Chromium browser instance for debugging, connect to it using Playwright 
    # over the Chrome DevTools Protocol (CDP), and create an instance of BrowserServerClient. 
    # The method accepts an optional user data directory for the browser and returns a new 
    # instance of BrowserServerClient with the initialized components.
    @classmethod
    async def Create(cls, UserDataDir = None):
        # Import the Playwright async API to enable asynchronous browser automation 
        # and interaction with the Chromium browser instance.
        from playwright.async_api import async_playwright

        # Launch a Chromium browser instance for debugging as a headless and detached process.
        # Retrive the process handle and the URL for connecting to the browser over CDP.
        Process, URL, _ = await LaunchChromeforDebugging(
            UserDataDir = UserDataDir, Headless = True, Detached = True
        )
        # Start the Playwright instance
        Playwright      = await async_playwright().start()
        # Connect to the launched browser instance over CDP using the retrieved URL.
        Browser         = await Playwright.chromium.connect_over_cdp(URL)
        # Create and return a new instance of BrowserServerClient.
        return cls(Playwright, Browser, Process, DeepSeekBrowserClient(Browser, False))

    # Delegate chat completion requests to the underlying browser client directly.
    # This method just forwards the parameters to the DeepSeekBrowserClient.
    async def ChatCompletions(self, Params):
        return await self.Client.ChatCompletions(Params)
    
    # Upload a file to DeepSeek through the browser client. The method first resolves 
    # the upload source into a byte stream, then calls the underlying browser client 
    # to perform the upload operation.
    async def UploadFile(self, Source, Filename, ModelType = "default"):
        return await self.Client.UploadFile(
            await _SourcetoBytesStream(Source), Filename, ModelType
        )
        
    # Delete a specified session by its session ID using the underlying browser client.
    async def DeleteSession(self, SessionID):
        Deleted = await self.Client.DeleteSession(SessionID)
        # If the deletion fails, a RuntimeError will be raised to indicate the failure.
        if not Deleted:
            raise RuntimeError(f"Failed to delete session: {SessionID}")

    # Close the browser and Playwright instances, then terminate the browser process 
    # if it is still running. This ensures that all resources are cleaned up properly.
    async def Close(self):
        try:
            await self.Browser.close()
            await self.Playwright.stop()
        finally:
            if self.Process.poll() is None:
                self.Process.terminate()

# Create a server client instance based on the specified mode (API or Browser).
# If BrowserMode is True, a BrowserServerClient instance will be created; 
# otherwise, an ApiServerClient instance will be created.
async def CreateServerClient(CredentialPath, BrowserMode = False, UserDataDir = None):
    if BrowserMode:
        return await BrowserServerClient.Create(UserDataDir)
    
    return await ApiServerClient.Create(CredentialPath)
