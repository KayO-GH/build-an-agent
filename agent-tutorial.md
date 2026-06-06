
# How to Build a Python AI Agent (with HuggingFace Inference API & Local Tools)

>This guide is inspired by [How to Build an Agent](https://ampcode.com/notes/how-to-build-an-agent) by ampcode.com. The code is unchanged from the original, but the explanations and commentary follow the step-by-step, annotated style of the reference blog post.

---

## 1. Why Build an Agent?

Modern LLMs can do more than just chat—they can use real tools! In this walkthrough, you’ll build a Python agent that:

- Chats using an open-source LLM (Qwen2.5-7B-Instruct) via the HuggingFace Inference API (no local model download required)
- Reads, lists, and edits files on your computer by detecting tool calls in the model’s responses
- Seamlessly combines LLM reasoning with real code execution

---

## 2. Prerequisites

Before you start, make sure you have:

- Python 3.9 or newer
- [uv](https://github.com/astral-sh/uv) for fast virtualenv and dependency management
- A HuggingFace account and a User Access Token with "Read" permission ([create one here](https://huggingface.co/settings/tokens))

---

## 3. Project Setup (Step-by-Step)

Let’s get your environment ready:

```bash
mkdir build-an-agent
cd build-an-agent
uv venv .venv
source .venv/bin/activate
uv pip install openai python-dotenv
```

Now, create a `.env` file in your project root with your HuggingFace token:

```
HF_TOKEN=your_huggingface_token_here
```

---

## 4. The Agent Code (How It Works)


Create a file named `agent.py` and paste in the following code. We’ll break down what each part does below.

```python
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

client = OpenAI(
  base_url="https://router.huggingface.co/v1",
  api_key=os.environ["HF_TOKEN"],
)

TOOLS = {
  "read_file": {
    "description": "Read the contents of a given file path.",
    "function": lambda path: open(path, "r").read() if os.path.exists(path) else "File not found."
  },
  "list_files": {
    "description": "List files and directories at a given path.",
    "function": lambda path=".": "\n".join(os.listdir(path))
  },
  "edit_file": {
    "description": "Replace 'old_str' with 'new_str' in the given file.",
    "function": lambda path, old_str, new_str: edit_file(path, old_str, new_str)
  }
}

def edit_file(path, old_str, new_str):
  if not os.path.exists(path):
    with open(path, "w") as f:
      f.write(new_str)
    return f"Created file {path}"
  with open(path, "r") as f:
    content = f.read()
  if old_str not in content:
    return "old_str not found in file"
  content = content.replace(old_str, new_str, 1)
  with open(path, "w") as f:
    f.write(content)
  return "OK"

def parse_tool_call(text):
  match = re.match(r"(\w+)\((.*?)\)", text)
  if not match:
    return None, None
  tool, args = match.groups()
  args = [a.strip().strip('"').strip("'") for a in args.split(",")]
  return tool, args

def main():
  print("Chat with your agent (Ctrl+C to quit)")
  history = [
    {
      "role": "system",
      "content": (
        "You are an AI assistant with access to the following tools: "
        "read_file(path), list_files(path), edit_file(path, old_str, new_str). "
        "If you need to use a tool, reply ONLY with the tool call in the format: tool_name(args). "
        "Do not explain, just output the tool call. Wait for the tool result before continuing the conversation."
      )
    }
  ]
  while True:
    user = input("\033[96mYou\033[0m: ")
    history.append({"role": "user", "content": user})
    messages = []
    for turn in history:
      if turn["role"] == "tool":
        messages.append({"role": "assistant", "content": turn["content"]})
      else:
        messages.append({"role": turn["role"], "content": turn["content"]})
    completion = client.chat.completions.create(
      model="Qwen/Qwen2.5-7B-Instruct:together",
      messages=messages,
    )
    response = completion.choices[0].message.content.strip()
    print(f"\033[93mAgent\033[0m: {response}")
    tool, args = parse_tool_call(response)
    if tool in TOOLS:
      result = TOOLS[tool]["function"](*args)
      print(f"\033[92mtool\033[0m: {tool}({', '.join(args)}) -> {result}")
      history.append({"role": "tool", "content": result})
    else:
      history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
  main()
```

### Code Commentary

- **Environment Setup:**
  - Loads your HuggingFace token from `.env` using `python-dotenv`.
  - Sets up the OpenAI client to use HuggingFace’s router endpoint.

- **Tool Definitions:**
  - `read_file(path)`: Reads a file’s contents.
  - `list_files(path)`: Lists files and directories at a path (defaults to current directory).
  - `edit_file(path, old_str, new_str)`: Replaces the first occurrence of `old_str` with `new_str` in a file, or creates the file if it doesn’t exist.

- **Tool Call Parsing:**
  - The agent looks for responses like `read_file("agent.py")` and parses out the tool name and arguments.

- **Main Loop:**
  - You chat with the agent in your terminal.
  - If the model wants to use a tool, it outputs only the tool call (no explanation).
  - The Python code detects this, runs the tool, and sends the result back as the next message.
  - The loop continues, allowing the agent to combine LLM reasoning with real code execution.

---

## 5. Example Conversation

Here’s what it looks like in action:

```
You: What files are in this directory?
Agent: list_files()
tool: list_files() -> agent.py\nREADME.md\n.env
You: What’s in agent.py?
Agent: read_file("agent.py")
tool: read_file(agent.py) -> [file contents]
You: Replace "foo" with "bar" in agent.py
Agent: edit_file("agent.py", "foo", "bar")
tool: edit_file(agent.py, foo, bar) -> OK
```

---

## 6. Extending Your Agent

- Add more tools by extending the `TOOLS` dictionary.
- For more reliable tool use, experiment with the system prompt and model choice.
- Always keep your HuggingFace token secure and never share it publicly.

---

## 7. Summary

You now have a Python agent that can chat, read, list, and edit files using an open LLM and real code execution—all in about 100 lines of code!

---

**Attribution:** This tutorial’s format and commentary are inspired by [How to Build an Agent](https://ampcode.com/notes/how-to-build-an-agent) by ampcode.com. The code is unchanged from the original tutorial.
