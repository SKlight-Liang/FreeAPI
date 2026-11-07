"""
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module provides an asynchronous Playwright-based client for interacting with
DeepSeek Chat through an existing browser session. It is mainly used for 
switching chat sessions, uploading files, deleting sessions, and submitting chat
completion requests while observing the corresponding browser network responses.

The implementation preserves the original request payload rules, response
matching conditions, timeout values, callback ordering, and browser interactions.
"""

import json
import asyncio

from .Utils import *

from playwright.async_api import Error

"""
DeepSeekBrowserClient Class

The class encapsulates the browser-based interactions with DeepSeek Chat.
It will manage the browser reference, verbosity setting, an asynchronous operation lock and a cached page.
The browser operations will be serialized through the lock to 
ensure asynchronous operation serialization and consistency.

Main workflow:
1. Reuse an existing DeepSeek page or create a new page.
2. Navigate between chat sessions when requested.
3. Observe upload, deletion, and completion network responses.
4. Return the same values and asynchronous data shape as the original code.

Function Table(Public):
- GetPage(): Retrieve or create a page object pointing to DeepSeek Chat.
- SwitchSession(SessionID): Switch to a specific chat session and return the session ID.
- UploadFile(Data, Filename, ModelType): Upload a file to DeepSeek and return the file ID.
- DeleteSession(SessionID): Delete a chat session and return the success status.
- ChatCompletions(Params): Submit a chat completion request and return the session ID and SSE stream generator.
"""

class DeepSeekBrowserClient:

    # Initialize the client, including saving the browser object and log switch,
    # creating a concurrency lock for serialized operations, and preparing a cached page reference.
    def __init__(self, Browser, Verbose = True):
        self.Browser = Browser
        self.Verbose = Verbose
        self._Lock   = asyncio.Lock()
        self._Page   = None

    # Get or create a page object pointing to DeepSeek Chat.
    # Here we will reuse the cached page and existing browser context if available, 
    # or create a new context and page when necessary.
    async def GetPage(self):
        # If the page has been cached, return it directly.
        if self._Page:
            return self._Page

        # Retrieve the first browser context if it exists, or just create a new one.
        if self.Browser.contexts:
            Context = self.Browser.contexts[0]
        else:
            Context = await self.Browser.new_context()

        # Look for an existing page with "deepseek.com" in the context
        self._Page = next(
            (Page for Page in Context.pages if "deepseek.com" in Page.url),
            None,
        )
        # If no such page exists, create a new one in the context.
        if not self._Page:
            self._Page = await Context.new_page()

        return self._Page

    # Switch the browser to a specific chat session and extract the session ID from the resulting URL.
    # This method uses a lock to ensure that only one session switch operation could be performed at a time, 
    # preventing race conditions and ensuring consistent state.
    async def SwitchSession(self, SessionID = ""):
        # Obtain the concurrent operation lock and 
        # perform the internal switching logic serially to avoid page state competition
        async with self._Lock:
            return await self._SwitchSession(SessionID)

    # The internal method for switching sessions without locking,
    # which navigates to the specified session URL and extracts the session ID from the final URL.
    async def _SwitchSession(self, SessionID = ""):
        # Get the page object
        Page = await self.GetPage()
        # Construct the target URL based on the provided session ID
        URL = ""
        if SessionID and len(SessionID) > 10:
            URL = f"https://chat.deepseek.com/a/chat/s/{SessionID}"
        else:
            URL = "https://chat.deepseek.com"

        # print(f"Switching to session: {SessionID} -> {URL}")

        # Navigate to the target URL and wait for the DOM content to load
        # await Page.goto(URL, wait_until = "domcontentloaded")
        try:
            # Use the commit policy to return immediately after the response header is received
            # to avoid waiting for JS to jump.
            await Page.goto(URL, wait_until = "commit")
        except Error as Exc:
            # If the competition is still caused by extremely fast jump, 
            # and the error message is clearly navigation interruption,
            # the safety will be ignored.
            if "interrupted by another navigation" not in str(Exc).lower():
                # Otherwise, we will raise the exception.
                raise Exc

        # Wait for a short duration to ensure the page is fully stabilized
        await Page.wait_for_timeout(1000)

        # Extract the session ID from the final URL if it contains the expected path
        if "/a/chat/s/" in Page.url:
            return Page.url.split("/a/chat/s/")[1].split("?")[0]

        return ""

    # Upload a file to the website with concurrent lock to avoid race conditions.
    # This method is driven by two response listeners:
    # The first one listens to the upload response to capture the file ID and status, 
    # while the second one listens for the processing status of the uploaded file.
    # Finally, it returns the file ID if the upload and processing are successful, or raises an error otherwise.
    async def UploadFile(self, Data, Filename, ModelType = "default"):
        # Acquire the concurrent lock to ensure that 
        # the upload process does not stagger with other page operations.
        async with self._Lock:
            # Get the page object
            Page = await self.GetPage()

            # If the current page is not on deepseek.com, navigate to the main chat page
            if "deepseek.com" not in Page.url:
                await Page.goto("https://chat.deepseek.com")

            # Get the current event loop and create a future to capture 
            # the file ID and status in the upload response
            Loop   = asyncio.get_running_loop()
            Future = Loop.create_future()

            # UPLOAD RESPONSE LISTENER: 
            # Captures the upload response and extracts the file ID 
            # and status if it matches the requested filename.
            async def Handle(Response):
                # Check if the response URL matches the upload endpoint and is successful
                if "/api/v0/file/upload_file" not in Response.url or not Response.ok:
                    return

                try:
                    # Parse the response JSON and extract the business data
                    ResponseData = await Response.json()
                    BusinessData = ((ResponseData.get("data") or {}).get("biz_data") or {})

                    # If the filename matches and the future is not already done, set the result
                    if BusinessData.get("file_name") == Filename and not Future.done():
                        Future.set_result(
                            (BusinessData.get("id"), BusinessData.get("status"))
                        )

                except Exception:
                    pass

            # Register the response listener for the upload response
            Page.on("response", Handle)

            try:
                # Set the file to upload through the file input element
                # with provided data, filename, and detected MIME type
                await Page.locator('input[type="file"][multiple]').set_input_files({
                    "name"    : Filename,
                    "buffer"  : bytes(Data),
                    "mimeType": DetectMIMEType(bytes(Data), Filename),
                })

                # Wait for the upload response to be captured within a timeout
                FileID, Status = await asyncio.wait_for(Future, 6)

                # If the file ID is not obtained, raise an error
                if not FileID:
                    raise RuntimeError("Failed to get fileId from upload response")
                # If the upload status is successful, return the file ID
                if Status == "SUCCESS":
                    return FileID
                # If the upload status indicates failure or empty content, raise an error
                if Status in ("FAILED", "CONTENT_EMPTY"):
                    raise RuntimeError(f"File upload failed for {Filename}: {Status}")

                # Otherwise, we need to enter the waiting phase 
                # for the server-side processing of the uploaded file.

                # Create a future to capture the processing status of the uploaded file
                Done = Loop.create_future()

                # RESOLUTION STATUS POLLING RESPONSE LISTENER:
                # Captures the processing status of the uploaded file by listening to
                # the fetch_files endpoint and checking the file ID and status.
                # The listener will set the result of the Done future
                # when the file is found in the response and its status is no longer "PARSING".
                async def Fetch(Response):
                    # Check if the response URL matches the fetch_files endpoint and contains the file ID
                    if "fetch_files" not in Response.url or FileID not in Response.url or not Response.ok:
                        return

                    try:
                        # Parse the response JSON and extract the business data
                        ResponseData = await Response.json()
                        Files = ((ResponseData.get("data") or {}).get("biz_data") or {}).get("files", [])

                        # Iterate through the files in the response to find the matching file ID
                        for File in Files:
                            if File.get("id") == FileID and File.get("status") != "PARSING" and not Done.done():
                                # Set the result of the Done future with the processing status
                                Done.set_result(File.get("status"))

                    except Exception:
                        pass

                # Register the response listener for the processing status
                Page.on("response", Fetch)

                try:
                    # Wait for the processing status to be captured within a timeout
                    ProcessingStatus = await asyncio.wait_for(Done, 60)
                finally:
                    # Remove the processing status listener finally to ensure that 
                    # it is cleaned up regardless of success or failure
                    Page.remove_listener("response", Fetch)

                # If the processing status is not successful, raise an error
                if ProcessingStatus != "SUCCESS":
                    raise RuntimeError(f"File processing failed for {Filename}: {ProcessingStatus}")

                return FileID

            finally:
                # Remove the upload response listener finally to ensure that 
                # it is cleaned up regardless of success or failure
                Page.remove_listener("response", Handle)

    # Delete a chat session by clicking the delete button on the page and observing the network response.
    # Same as above, this method also uses a lock to ensure that the deletion operation is serialized
    # and does not interfere with other page operations.
    async def DeleteSession(self, SessionID):
        # Obtain the concurrent operation lock
        async with self._Lock:
            # Get the page object
            Page = await self.GetPage()
            # Create a future to capture the result of the delete request
            Future = asyncio.get_running_loop().create_future()

            # DELETION RESPONSE LISTENER:
            # Captures the response of the delete request and sets the result of the future
            async def OnResponse(Response):
                # Check if the response URL matches the delete endpoint and the future is not already done
                if "/api/v0/chat_session/delete" in Response.url and not Future.done():
                    # Set the result of the future with the response status (True for success, False for failure)
                    Future.set_result(Response.ok)

            # Register the response listener for the delete request
            Page.on("response", OnResponse)

            try:
                # Execute the page-side JavaScript to click the delete button for the specified session ID.
                # The JavaScript code will searche for the delete button in the DOM and clicks it,
                # then checks for the confirmation dialog and clicks the confirm button if found.
                Result = await Page.evaluate('''
                    async id => {
                        const t = document.querySelector(`[href="/a/chat/s/${id}"]`)
                                    ?.lastElementChild
                                    ?.querySelector('[role="button"]'); 
                        if (!t) return null;

                        t.click();
                        await new Promise(r => setTimeout(r, 300));

                        let found = false;
                        for (const o of document.querySelectorAll('.ds-dropdown-menu-option__label')) {    
                            if (o.textContent?.includes('删除')) {
                                o.click();
                                found = true;
                                await new Promise(r => setTimeout(r, 300));
                                break;
                            }
                        }
                        if (!found) return false;

                        const b = Array.from(
                            document.querySelectorAll('[role="button"].ds-button--error')
                        ).find(x => x.textContent?.trim() === '删除该对话');
                        if (!b) return false;
                        
                        b.click();
                        return true;
                    }
                ''',SessionID,)

                # If the JavaScript execution did not find the delete button,
                # return the result (None or False)
                if not Result:
                    return Result

                # Wait for the delete request response to be captured within a timeout
                try:
                    return await asyncio.wait_for(Future, 3)
                except asyncio.TimeoutError:
                    return False

            finally:
                # Remove the delete response listener finally to ensure that 
                # it is cleaned up regardless of success or failure
                Page.remove_listener("response", OnResponse)

    # Initiate a chat completion request by hijacking the request supplementary parameters.
    # We will intercept the request to inject the session ID, model type, and other optional parameters,
    # and then grab the SSE stream response to yield the raw data chunks.
    # The method returns the captured session ID and an asynchronous generator for the SSE stream.
    async def ChatCompletions(self, Params):
        # Acquire the concurrent operation lock
        async with self._Lock:
            # Get the page object
            Page = await self.GetPage()
            # Switch to the specified session ID and capture the resulting session ID
            SessionID = await self._SwitchSession(Params.get("session_id", ""))
            CapturedSessionID = SessionID
            # Define the request pattern to intercept for chat completions
            Pattern = "**/api/v0/chat/completion"

            # ROUTE HANDLER:
            # Intercept the matching completion request and modify its payload before it is sent.
            async def RouteHandler(Route):
                nonlocal CapturedSessionID

                # Get the request object from the route
                Request = Route.request
                # Try to parse the request payload as JSON, defaulting to an empty dictionary if parsing fails
                try:
                    Payload = json.loads(Request.post_data or "{}")
                except:
                    Payload = {}

                # Extract the prompt from the request payload and the wanted message from the parameters
                Prompt = str(Payload.get("prompt", ""))
                Wanted = Params["message"]
                # Compare the first character of the prompt and the wanted message to 
                # determine whether it is the expected request to intercept. 
                # If not, continue the route without modification.
                if not Prompt or not Wanted or Prompt.lstrip()[:1] != Wanted.lstrip()[:1]:
                    return await Route.continue_()
                
                # If the prompt matches, inject the session ID and model type into the request payload.
                Payload["chat_session_id"] = Payload.get("chat_session_id") or SessionID
                Payload["model_type"]      = Params.get("model_type", "default")
                # Validate the session ID to ensure it is a valid string with sufficient length.
                if not isinstance(Payload["chat_session_id"], str) or len(Payload["chat_session_id"]) <= 10:
                    raise RuntimeError("Failed to capture session ID from request payload")
                # Update the captured session ID with the one from the request payload.
                CapturedSessionID = Payload["chat_session_id"]

                # Map the optional parameters from the input Params to the request payload,
                # only if they are explicitly provided and not None.
                ParameterMappings = [
                    ("file_ids"         , "ref_file_ids"),
                    ("search_enabled"   , "search_enabled"),
                    ("thinking_enabled" , "thinking_enabled"),
                    ("parent_message_id", "parent_message_id"),
                    ("preempt"          , "preempt"),
                ]
                # Iterate through the parameter mappings and copy the values.
                for SourceKey, DestinationKey in ParameterMappings:
                    if SourceKey in Params and Params[SourceKey] is not None:
                        Payload[DestinationKey] = Params[SourceKey]

                # Continue the route with the modified payload, ensuring it is serialized as JSON.
                await Route.continue_(post_data=json.dumps(Payload, ensure_ascii=False))
                # Unregister the route handler after the request has been modified and sent.
                await Page.unroute(Pattern, RouteHandler)

            # Register the route handler to intercept the chat completion request.
            await Page.route(Pattern, RouteHandler)

            try:
                # Fill the chat input textarea with the message from the parameters.
                await Page.locator('textarea[autocomplete="off"]').fill(Params["message"])

                # Create a task to wait for the response of the chat completion request.
                ResponseTask = asyncio.create_task(
                    Page.wait_for_event(
                        "response",
                        predicate = lambda Response: (
                            "/api/v0/chat/completion" in Response.url
                        ),
                    )
                )

                try:
                    # Attempt to click the file input button if it exists
                    Target = Page.locator('input[type="file"][multiple] + *').first
                    if await Target.count():
                        await Target.click()
                    else:
                        # Otherwise, we need to simulate pressing the Enter key to submit the chat message
                        await Page.keyboard.press("Enter")
                except Exception:
                    await Page.keyboard.press("Enter")

                # Wait for the response of the chat completion request.
                Response = await ResponseTask
                # Get the content type of the response headers.
                ContentType = (await Response.all_headers()).get("content-type", "")
                # Read the raw response text.
                RawResponse = await Response.text()

                # Validate that the response is an SSE stream, otherwise raise an error.
                if "text/event-stream" not in ContentType:
                    raise RuntimeError(
                        f"Unexpected content type[{ContentType}] for chat completion response: {RawResponse}"
                    )

                # ASYNCHRONOUS GENERATOR:
                # Provides original response bytes for upper layers to consume as an SSE stream.
                async def Generate():
                    yield RawResponse.encode()

                # Return the captured session ID and the asynchronous generator for the SSE stream.
                return CapturedSessionID, Generate()

            finally:
                # Ensure that the route handler is unregistered after the request is completed,
                # even if an exception occurs during the process.
                try:
                    await Page.unroute(Pattern, RouteHandler)
                except:
                    pass
