<div align="center">

# FreeAPI

### From browser conversations to programmable interfaces.

**A browser-backed integration layer for LLMs, with OpenAI-style request and response normalization.**

`Browser automation` · `Streaming SSE` · `Tool calling` · `Session management`

[Overview](#overview) · [Getting started](#getting-started) · [License](#license)

Version: **Beta 0.3.0**(Developing)

</div>

---

**LLM APIs cost a fortune?**  
**Your Plus subscription limits keep getting in the way?**

**What if I told you...** you could turn free AI chat websites into programmable workflows?

**Sounds interesting?** Meet **FreeAPI** Now!

## Overview

FreeAPI bridges the gap between the website chat and your favorite AI-powered applications. Instead of being limited to chatting in a browser, you can bring it into your own projects, tools, and workflows through a familiar API-style interface.

No complicated setup concepts. No need to understand what's happening behind the scenes. Just a simpler way to connect the tools you love with the AI you love. Everything is just like you are using an official API, of course, without any possible costs or limits.

So what can you do with FreeAPI?

- Connect your favorite tools: Bring web-based AI into applications and workflows that support compatible API interfaces.
- Keep the conversation flowing: Work with regular responses or real-time streaming.
- Go beyond simple chat: Explore tool calling and more advanced AI-powered workflows.
- Made for curious minds: Whether you're a developer, researcher, student, or simply an AI enthusiast, FreeAPI is designed to make experimentation more accessible.

All in all, you bring the ideas, and FreeAPI helps make the connection.

## Getting started

Ready to give FreeAPI a try? Let's get you up and running!

### Requirements

Before we begin, make sure you have:
- **Python 3.12 or newer** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)
- **Google Chrome** — For browser-based AI interactions

### Installation

First, clone the repository:

```bash
git clone https://github.com/SKlight-Liang/FreeAPI.git
cd FreeAPI
```

Next, install FreeAPI and its dependencies:

```bash
pip install -e .
```

That's it! Python will take care of installing the required packages.

> [!TIP]
> We highly recommend using **a virtual environment** to avoid conflicts with other Python packages. What? You ask me how to create a virtual environment? Maybe you can turn to little blue whale for help. 

### Run

#### CLI (Browser Mode)

Now for the fun part! Start FreeAPI with:

```bash
FreeAPI-ChatB -i -d -p PORT
```

FreeAPI will launch its command-line entry point.

> [!NOTE]
> The `-i` flag enables interactive mode, allowing you to chat with the AI in multiple rounds. The `-d` flag means that the chat session will be deleted after completion. The `-p PORT` flag specifies the port number for the browser's CDP debugger (default is 9222). If you meet any issues, you can try changing the port number to avoid conflicts.

#### CLI (Headless Mode)

FreeAPI now supports a direct web-client workflow! Before starting a chat, obtain your DeepSeek authentication credentials using the browser-assisted authentication command:

```bash
FreeAPI-Auth -p PORT
```

Follow the browser login process when prompted. FreeAPI will capture the required credentials and save them locally for subsequent requests. You generally only need to repeat this step when your credentials expire or become invalid.

> [!NOTE]
> Same as above, `FreeAPI-Auth` also uses Chromium-based browser automation, so you can change the port number with the `-p PORT` flag if you encounter any conflicts.

Once authentication is complete, launch the terminal chat client:

```bash
FreeAPI-Chat -i -d -t -s
```

This command starts FreeAPI in headless mode, allowing you to chat with the AI without opening a browser window. After that, you can chat with AI happily!

> [!NOTE]
> The `-i` flag enables interactive mode and the `-d` flag means that the chat session will be deleted after completion. In addition, we also provide two optional flags to help you catch a glimpse of how AI works behind the scenes. The `-t` flag enables the display of tool calls, while the `-s` flag enables much more details about the search process, including the search results and some other things. You can use these flags to gain a deeper understanding of how AI generates responses.

#### Server(Completion Request)

We have launched a brand new mode! Now you can use FreeAPI as an official API server to handle completion requests. Same as above, we also support both browser mode and headless mode. Therefore, you need to determine whether to run the authentication command first based on your mode of choice. After that, you can start the server with the following command:

```bash
FreeAPI-Server -p PORT
```

Then FreeAPI will start a local server to handle completion requests. 

> [!NOTE]
> The `-p PORT` flag specifies the port number for the server (default is 10086). You can change the port number to avoid conflicts if needed. The server will listen for incoming requests and respond with the AI-generated completions. Additionally, you can use the optional `--browser` flag to enable browser mode, which allows FreeAPI to interact with web-based AI services through a browser interface. If you don't use this flag, FreeAPI will run in headless mode by default.

After the server is running, you can visit the following URL in your web browser to check whether the server is running properly:

```
http://localhost:PORT/v1/models
```

If you could see a list of available models, congratulations! FreeAPI is running successfully. You can now send completion requests to the server using the appropriate API endpoints:

```
http://localhost:PORT/v1/chat/completions
```

### What's Next?

Once FreeAPI is running, you're ready to explore how to connect web-based AI services to your own applications and workflows.

Here we will provide an example of how to send requests to FreeAPI and receive responses. You can use any programming language or tool that supports HTTP requests, such as Python, JavaScript, or Postman.

```python
import json
import requests

# Set the completion endpoint URL
URL = "http://localhost:PORT/v1/chat/completions"

# Define the request payload
Messages = [{
    "role"    : "user",
    "content" : [{
        "type": "text",
        # The text content of the message
        "text": "Hello, FreeAPI! Can you tell me a joke?"
    }, {
        "type": "file",
        "file": {
            # The base64-encoded content of the file
            # or you can use the ABSOLUTE PATH of the file.
            "file_data": "SSBhbSBsaXR0bGUgYmx1ZSB3aGFsZS4=",
            "filename" : "Who am I.txt"
        }
    }, {
        "type"     : "image_url",
        "image_url": {
            # The ABSOLUTE PATH or URL of the image file.
            "url": r"\ReadmeSrc\Gamma 0.1.2 Architecture.svg"
        }
    }]
}]

# Define the tool call payload
Tools    = []

# Define the request payload
Payload = {
    "model"      : "deepseek",
    "messages"   : Messages,
    "tools"      : Tools,
    "tool_choice": "auto"
}

# Send the request to FreeAPI
Response = requests.post(URL, json = Payload)
# Parse the response
ResponseBody = Response.json()
Message      = ResponseBody["choices"][0]["message"]
ToolCalls    = Message.get("tool_calls", [])
```

> [!WARNING]
> Unlike the official API, FreeAPI only accepts absolute paths for file and image inputs. Please ensure that you provide the correct absolute path to the files you want to include in your requests. More details about how to use FreeAPI can be found in the [test code](./TestSrc/CompletionsTest.py).
## License

FreeAPI is an open-source software released under the [MIT License](./LICENSE). You're welcome to use, modify, and share it! Of course, nothing could be better if you could give us a star on GitHub and advertise it to your friends!

> [!IMPORTANT]
> We built FreeAPI to make AI more accessible, not to encourage abuse of free services. Please use this project responsibly, include but not limited to:
> - **Respect the platforms.** Follow the terms of service and usage policies of the AI providers you interact with.
> - **Be mindful of shared resources.** Avoid excessive automated requests or activities that could disrupt services for others.
> - **Play fair.** Don't use FreeAPI to bypass security measures, evade restrictions, or engage in malicious activities.
>
> **Let's keep AI accessible, sustainable, and enjoyable for everyone!**

>[!WARNING]
> We are not affilicated with any AI providers, and FreeAPI is not an official client. So we still insist on using official APIs for production or commercial purposes. FreeAPI is intended for personal use, experimentation, and learning. Please be aware of the risks and limitations when using it with third-party services. 
> 
> **Our team are not responsible for any consequences arising from its use, including account suspension, service disruption, or legal issues.**

---

<div align="center">

**FreeAPI** · Browser-native integration, interface-oriented design.

<sub>Independent project. Not an official client.</sub>

</div>
