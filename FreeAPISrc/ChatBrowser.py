"""
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module defines a command-line interface for interacting with
the DeepSeek web page through a Chromium-based browser in CDP debugging 
mode. It will connect to a local browser, create a DeepSeekBrowserClient, 
send messages and stream responses. The thinking content will be output 
to stderr, while the formal reply will be output to stdout.
"""

import sys
import asyncio
import argparse

from .Browser import *
from .DeepSeekBrowserClient import *
from .DeepSeekStreamParser import *

from playwright.async_api import async_playwright

# FEATURE: We optimize the resource recovery algorithm to ensure that 
# the browser process could be terminated properly in most cases. Here
# we use an elegant tow-stage termination strategy to ensure that the 
# browser process has been terminated gracefully and forcefully if necessary.
async def StopBrowserProcess(Process):
    # If the process is None or has already exited, return immediately.
    if Process is None or Process.poll() is not None:
        return

    # First, attempt to terminate the process gracefully.
    Process.terminate()

    try:
        # Wait for the process to exit within 5 seconds.
        await asyncio.wait_for(asyncio.to_thread(Process.wait), timeout = 5)
        return
    except asyncio.TimeoutError:
        pass

    # If the process is still running after the timeout, forcefully kill it.
    if Process.poll() is None:
        Process.kill()

    await asyncio.to_thread(Process.wait)

# Parse command-line arguments and run the browser-backed chat workflow.
# Keep the original initialization order and cleanup boundaries unchanged.
# It realizes multiple rounds of dialogue with the model in an asynchronous manner.
async def _Main():
    # Create an argument parser
    Parser = argparse.ArgumentParser()
    # Add command-line arguments for:
    # Message: an optional initial message to send to the chat model.
    Parser.add_argument("message", nargs = "?")
    # Interactive: a flag to enable continuous chatting after the initial message.
    Parser.add_argument("-i", "--interactive", action = "store_true")
    # Delete: a flag to delete the chat session after completion.
    Parser.add_argument("-d", "--delete", action = "store_true")
    # CDPPort: an optional port number for the browser's CDP debugger.
    Parser.add_argument("-p", "--port", type = int, default = DEFAULT_CDP_PORT)
    # UserDataDir: an optional path to specify the browser's user data directory.
    Parser.add_argument("--user-data-dir")

    Arguments = Parser.parse_args()

    # Launch the browser debugging process and obtain the WebSocket URL for Playwright to connect.
    Process, URL, _ = await LaunchChromeforDebugging(
        UserDataDir = Arguments.user_data_dir,
        Headless    = False,
        Detached    = True,
        CDPPort     = Arguments.port
    )
    # Start Playwright and connect to the browser over CDP.
    Playwright = await async_playwright().start()
    Browser    = await Playwright.chromium.connect_over_cdp(URL)

    # Create a DeepSeekBrowserClient instance to handle chat interactions.
    Client = DeepSeekBrowserClient(Browser)
    # Initialize session and parent message identifiers to None for the first request.
    SessionID       = None
    ParentMessageID = None

    # Use a try-finally block to ensure proper cleanup of resources and optional session deletion.
    try:
        # Retrieve the first message from the command-line arguments
        Message = Arguments.message

        # Loop to handle multiple rounds of chat 
        # if there are messages or the interactive mode is enabled.
        while Message or Arguments.interactive:
            # If there is no ready-made message, 
            # read a line from standard input and remove the leading and trailing blanks.
            if not Message:
                Message = input("YOU > ").strip()
                # If the input is empty, continue to the next iteration and wait for input again.
                if not Message:
                    continue

            # If the message entered by the user is "exit", break the loop.
            if Message.lower() == "exit":
                break

            # Submit the message to the DeepSeekBrowserClient with the current session，
            # parent message identifiers, message text and model/search/think switches.
            SessionID, ResponseBody = await Client.ChatCompletions({
                "session_id"       : SessionID,
                "parent_message_id": ParentMessageID,
                "message"          : Message,
                "model_type"       : "default",
                "search_enabled"   : True,
                "thinking_enabled" : True,
            })
            # Parse the streamed response from the server to
            # get the thinking content, the formal reply or other important information.
            Result = await ParseStream(ResponseBody)

            # If the thinking content is present, print it to standard error.
            if Result["thinking"]:
                print("THINKING >\n" + Result["thinking"], file = sys.stderr)
            # Print the formal reply to standard output.
            print("REPLY >\n" + Result["text"])

            # Update the parent message identifier for the next request.
            ParentMessageID = Result["message_id"]
            # Clear the message variable to indicate that the next message should be read from input.
            Message = None

            # If the interactive mode is not enabled, break the loop after one round of chat.
            if not Arguments.interactive:
                break

    finally:
        # If the delete flag is set and a session was created, attempt to delete the session.
        if Arguments.delete and SessionID:
            try:
                await Client.DeleteSession(SessionID)
            except Exception:
                # Suppress any exceptions during session deletion to avoid crashing the program.
                pass

        # Close the browser and stop Playwright in the original sequence.
        try:
            await Browser.close()
        except Exception:
            pass

        try:
            await Playwright.stop()
        except Exception:
            pass

        # If the browser process is still running, terminate it to free resources.
        await StopBrowserProcess(Process)
            
# Synchronous entry point for callers of the command-line module.
def main():
    asyncio.run(_Main())

# Backward-compatible aliases preserve potential external Python call sites.
_main = _Main
