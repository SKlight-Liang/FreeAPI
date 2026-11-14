'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module is a command line tool used to obtain authentication credentials 
(cookies, Bearer Tokens, User Agents) from the browser that has logged in to 
DeepSeek, and save it as a JSON file for reuse by other components. This 
module will connect to a Chromium debugging endpoint, wait for the user to 
log in and collect the browser credentials.
'''

import os
import sys
import json
import time
import shutil
import asyncio
import argparse
import subprocess
import urllib.request

from pathlib import Path
from playwright.async_api import async_playwright

from .Utils import *

# Default CDP remote debugging port for Chromium-based browsers.
DEFAULT_PORT = 9222

# Return the default credential file path relative to the current working directory.
def DefaultCredentialPath():
    return str(DefaultDataDir() / "DeepSeekCredentials.json")

# Return the default user data directory path for the browser profile.
def DefaultUserDataDir():
    return str(DefaultDataDir() / "Profile-UserData")

# Save the provided credentials as a JSON file at the specified path.
def SaveCredentials(Credentials, PathName):
    # Create the destination directory
    Path(PathName).parent.mkdir(parents = True, exist_ok = True)
    # Serialize credentials to JSON with UTF-8 encoding
    Path(PathName).write_text(
        json.dumps(Credentials, ensure_ascii = False, indent = 2) + "\n", 
        encoding = "utf8"
    )

# Read the credential dictionary from a specified JSON file
def LoadCredentials(PathName):
    return json.loads(Path(PathName).read_text(encoding = "utf8"))

# Poll the debugging endpoint until its version document reports a WebSocket URL.
# That means the browser is ready to accept CDP connections.
async def WaitforDebugger(CDPURL = None, Timeout = 20):
    # Calculate the deadline for waiting 
    EndTime = time.time() + Timeout

    # Loop trying to access the debugging interface before the timeout expires
    while time.time() < EndTime:
        try:
            # Attempt to request the version endpoint. 
            with urllib.request.urlopen(CDPURL + "/json/version", timeout = 2) as Response:
                ResponseData = json.load(Response)
                # If we could successfully obtain the websocket debugger URL, 
                # the browser is ready to be connected.
                if ResponseData.get("webSocketDebuggerUrl"):
                    return

        except Exception:
            # Ignore any exception of connection or resolution, just continue to retry
            pass

        # Wait for a short interval before the next attempt.
        await asyncio.sleep(0.5)

    # If the timeout expires without finding a WebSocket debugger URL,
    # raise a TimeoutError to indicate that the browser is not ready for debugging.
    raise TimeoutError(f"Timed out waiting for browser debugger at {CDPURL}")

# Detect the browser executable path based on the platform and environment variables.
def FindBrowserExecutable():
    # Honor the explicitly configured browser executable.
    Configured = os.environ.get("FREEAPI_BROWSER")
    if Configured:
        return Configured

    # Detect browser installation paths for each platform.
    # Use default paths for Windows, macOS, and Linux to 
    # find the first available browser executable.
    if sys.platform == "win32":
        Candidates = [
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        ]
    elif sys.platform == "darwin":
        Candidates = [
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path.home() / "Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ]
    else:
        Candidates = [
            Path("/usr/bin/microsoft-edge"),
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/chromium"),
        ]
    # Check each candidate path and return the first one that exists as the browser executable.
    for Candidate in Candidates:
        if Candidate.is_file():
            return str(Candidate)

    # If no candidate paths were found, try to locate the browser executable in the system PATH.
    for Name in ("microsoft-edge", "google-chrome", "chromium"):
        Executable = shutil.which(Name)
        if Executable:
            return Executable

    raise FileNotFoundError("No supported Chromium-based browser was found.")

# Start the browser process in remote debugging mode and return the Popen process object.
def LaunchBrowser(Port = DEFAULT_PORT, UserDataDir = None):
    # Find an avaliable Chromium-based browser executable through specific programs.
    BrowserExecutable = FindBrowserExecutable()
    # Determine the user data directory and ensure its existence.
    ProfileDir = UserDataDir or DefaultUserDataDir()
    Path(ProfileDir).mkdir(parents = True, exist_ok = True)

    # Build the command line arguments to launch the browser with CDP enabled.
    Arguments = [
        BrowserExecutable,
        f"--remote-debugging-port={Port}",
        f"--user-data-dir={ProfileDir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync"
    ]

    # Launch the browser process with the specified remote debugging port and 
    # user data directory. Here we will discard the stdout and stderr streams 
    # to avoid cluttering the console with browser logs.
    return subprocess.Popen(
        Arguments,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )

# Connect to an opened browser and wait for a DeepSeek login token.
# We will try to capture the cookies, Bearer Token, and User Agent 
# from the browser context after the user logs in to the DeepSeek.
async def Auth(CDPPort = None, AutoLaunch = True, UserDataDir = None, Keep = False):
    URL            = None
    Port           = None
    BrowserProcess = None
    
    # Standardize the CDP endpoint URL based on the provided port.
    # Or we just use the default port if no CDPPort is provided.
    if isinstance(CDPPort, int) or (isinstance(CDPPort, str) and CDPPort.isdigit()):
        Port = int(CDPPort)
    else:
        Port = DEFAULT_PORT
    URL = f"http://127.0.0.1:{Port}"

    # In most cases, we need to launch a new browser process to capture the credentials. 
    # However, it is also possible that the user has already launched a browser with CDP 
    # enabled. This situation may occur if FindBrowserExecutable() is unable to locate 
    # a browser executable, user should manually perform this step.
    if AutoLaunch:
        BrowserProcess = LaunchBrowser(Port, UserDataDir)
    # Wait for the debugging port to be ready for accepting connections.
    await WaitforDebugger(URL)

    # Start the Playwright context 
    async with async_playwright() as Playwright:
        # Connect to the browser using the Chrome DevTools Protocol (CDP) endpoint.
        Browser = await Playwright.chromium.connect_over_cdp(URL)
        # Get the first available browser context from the connected browser.
        Context = Browser.contexts[0]
        # Find an existing page in the context that has a URL containing "deepseek.com".
        Page = None
        for Candidate in Context.pages:
            if "deepseek.com" in Candidate.url:
                Page = Candidate
                break
        # If no existing page is found, create a new page in the context.
        if not Page:
            Page = await Context.new_page()
        # Navigate to the DeepSeek chat page and wait for the DOM content to be loaded.
        await Page.goto("https://chat.deepseek.com", wait_until = "domcontentloaded")
        
        Token  = None
        Prompt = False
        # Repeatedly check the localStorage for the userToken until it is available.
        while not Token:
            try:
                # Retrieve userToken from localStorage
                RowToken = await Page.evaluate("localStorage.getItem('userToken')")
                # If the token is found, parse it as JSON and extract the value field.
                if RowToken:
                    Token = json.loads(RowToken).get("value")
            except Exception:
                pass

            # If we have not obtained the token yet, prompt the user to log in
            if not Token:
                if not Prompt:
                    Prompt = True
                    print("AUTH > Please finish logging into DeepSeek in the browser window.")
                # Wait for a short interval before checking again.
                await asyncio.sleep(1)

        # Obtain the cookies and user agent from the browser context after successful login.
        Cookies     = await Context.cookies("https://chat.deepseek.com")
        UserAgent   = await Page.evaluate("navigator.userAgent")
        # Format the cookies into a single string suitable for HTTP headers.
        Cookie      = "; ".join(f"{Cookie['name']}={Cookie['value']}" for Cookie in Cookies)
        # Construct the credentials dictionary containing the cookie string, Bearer Token, and User Agent.
        Credentials = {
            "cookie"   : Cookie,
            "bearer"   : Token,
            "userAgent": UserAgent,
        }

        # If the browser was launched by this script, or the user has not requested to 
        # keep it running, then we will terminate the browser process to clean up resources. 
        # Otherwise, we leave the browser running for further use.
        if BrowserProcess and not Keep:
            BrowserProcess.terminate()

        return Credentials

# Parse command-line arguments and run the authentication workflow.
# Persist the obtained credentials to a spsecific path.
async def _Main():
    # Create an argument parser
    Parser = argparse.ArgumentParser()
    # Add command-line arguments for:
    # Output: the path to save the obtained credentials JSON file.
    Parser.add_argument("-o", "--output", default = DefaultCredentialPath())
    # AutoLaunch: a flag to indicate whether to launch a new browser process.
    # Parser.add_argument("-l", "--autolaunch", action = "store_true")
    # Keep: a flag to indicate whether to keep the browser process 
    # running after obtaining credentials.
    Parser.add_argument("-k", "--keep", action = "store_true")
    # CDPPort: an optional port number for the browser's CDP debugger.
    Parser.add_argument("-p", "--port", type = int, default = DEFAULT_PORT)
    # UserDataDir: an optional path to specify the browser's user data directory.
    Parser.add_argument("--user-data-dir")

    Arguments = Parser.parse_args()

    # Run the authentication workflow to obtain credentials from the browser.
    Credentials = await Auth(
        CDPPort     = Arguments.port,
        # AutoLaunch  = Arguments.autolaunch,
        UserDataDir = Arguments.user_data_dir,
        Keep        = Arguments.keep
    )
    # Save the obtained credentials to the specified output path as a JSON file.
    SaveCredentials(Credentials, Arguments.output)
    print(f"AUTH > Saved credentials to {Arguments.output}")

# Synchronous entry point for callers of the command-line module.
def main():
    asyncio.run(_Main())

# Backward-compatible aliases preserve potential external Python call sites.
_main = _Main
