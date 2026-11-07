"""
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module provides a simple interface to launch a Chromium-based browser 
in CDP(Chrome DevTools Protocol) mode for debugging and automation purposes
and wait for the debugging endpoint to become available.
It allows users to automatically control the local Edge/Chrome browser
for upper browser client connection. (We use port 9222 by default.)
The module will build default debugging locations, checks the browser's version
endpoint for a WebSocket debugger URL, and launches an Edge-compatible browser.

Sadly, this module is not cross-platform. It is only tested on Windows 10/11 with Edge installed.
Hope that in the future, we can make it cross-platform and much more convenient to use.
After all, the execution speed of `playwright install chromium` is really TOO SLOW !!!!!!

Functions Table:
- DefaultCDPURL(CDPPort = 9222):
Return the CDP debugging address for the specified port.
- DefaultUserDataDir():
Return the default user data directory path for the browser profile: .data/profile-9222
- WaitforDebuggerURL(CDPURL = None, Timeout = 20):
Poll the browser's version endpoint until a WebSocket debugger URL is found or the timeout expires.
- LaunchChromeforDebugging(CDPPort = 9222, UserDataDir = None, Headless = False, Detached = False):
Launch a Chromium-based browser process with CDP enabled and wait for its debugging endpoint to be ready
- FindBrowserExecutable():
Find the path to the browser executable based on the platform and environment variables.
"""

import os
import sys
import json
import time
import shutil
import asyncio
import subprocess
import urllib.request

from pathlib import Path

# Default CDP remote debugging port for Chromium-based browsers.
DEFAULT_CDP_PORT = 9222
# Default path to the Microsoft Edge executable on Windows.
# CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# FEATURE: Previous implementation was hardcoded to use Microsoft Edge, 
# so it was only compatible with Windows. Here we will try to make it more cross-platform 
# by detecting the browser executable path based on the platform and environment variables.
def FindBrowserExecutable():
    # Honor the explicitly configured browser executable.
    Configured = os.environ.get("FREEAPI_BROWSER")
    if Configured:
        return Configured

    # Detect browser installation paths for each platform.
    # Use default paths for Windows, macOS, and Linux to find the first available browser executable.
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

# Return the default local HTTP endpoint for the browser's CDP debugger.
def DefaultCDPURL(CDPPort = DEFAULT_CDP_PORT):
    return f"http://127.0.0.1:{CDPPort}"

# Return the default local directory for the browser's user data profile.
def DefaultUserDataDir():
    return str(Path.cwd() / ".Data" / f"Profile-UserData")

# Polling waiting for the browser's debugging endpoint to be ready.
async def WaitforDebuggerURL(CDPURL = None, Timeout = 20):
    # Determine the debugging address and deadline to poll.
    URL     = CDPURL or DefaultCDPURL()
    EndTime = time.time() + Timeout  

    # Loop attempting to read the version endpoint 
    # until a WebSocket debugger URL is found or the timeout expires.
    while time.time() < EndTime:
        try:
            # Request the version endpoint to retrieve the debugger information.
            with urllib.request.urlopen(URL + "/json/version", timeout = 2) as Response:
                ResponseData = json.load(Response)
                # If the response contains a WebSocket debugger URL, 
                # that means the browser is ready for debugging, so we can return it happly.
                if ResponseData.get("webSocketDebuggerUrl"):
                    return ResponseData["webSocketDebuggerUrl"]

        except Exception:
            # Ignore any exception from opening, reading, or parsing the endpoint, 
            # just continue to retry until the timeout expires.
            pass

        # If the response does not contain a WebSocket debugger URL,
        # we will wait for a short interval before trying again.
        await asyncio.sleep(0.5)

    # If the timeout expires without finding a WebSocket debugger URL,
    # we will raise a TimeoutError to indicate that the browser is not ready for debugging.
    raise TimeoutError(f"Timed out waiting for browser debugger at {URL}")

# Start a browser process with CDP enabled and wait for its debugging endpoint to be ready.
# Then return the process object, CDP URL and the user data directory path.
async def LaunchChromeforDebugging(CDPPort = DEFAULT_CDP_PORT, UserDataDir = None, Headless = False, Detached = False):
    # Confirm whether the user profile directory is exists.
    # If not, we need to create it before launching the browser.
    ProfileDir = UserDataDir or DefaultUserDataDir()
    Path(ProfileDir).mkdir(parents = True, exist_ok = True)

    # Use the specific function to find the browser executable.
    BrowserExecutable = FindBrowserExecutable()
    # Build the command line arguments to launch the browser with CDP enabled.
    Arguments = [
        BrowserExecutable,
        f"--remote-debugging-port={CDPPort}",
        f"--user-data-dir={ProfileDir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
    ]

    # If headless mode is requested, append the headless flags to the arguments list.
    if Headless:
        Arguments += ["--headless=new", "--hide-scrollbars"]
    # If detached mode is requested, set the process creation flags to create a new process group.
    # This allows the browser process to run independently of the parent process.
    # However, this feature is only available on Windows, 
    # other platforms will ignore this flag and run the browser normally.
    # FEATURE: We want to support detached mode not only on Windows, but also on Linux and macOS.
    CreationFlags = 0
    StartNewSession = False

    if Detached:
        if sys.platform == "win32":
            CreationFlags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            StartNewSession = True

    # Launch the browser process with the specified arguments and flags.
    BrowserProcess = subprocess.Popen(
        Arguments,
        stdout            = subprocess.DEVNULL,
        stderr            = subprocess.DEVNULL,
        creationflags     = CreationFlags,
        start_new_session = StartNewSession
    )

    # Construct the URL of CDP debugger endpoint and wait for it to be ready.
    URL = f"http://127.0.0.1:{CDPPort}"
    await WaitforDebuggerURL(URL, 20)

    return BrowserProcess, URL, ProfileDir
