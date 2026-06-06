# How to Build a Python AI Agent

> This guide is inspired by [How to Build an Agent](https://ampcode.com/notes/how-to-build-an-agent) by Thorsten Ball. The original builds a code-editing agent in Go with Anthropic's Claude. This version rebuilds the same idea in Python (because it's more popular) with the HuggingFace Inference API (because it's virtually free to start) and grows the agent in small stages.

The point of this tutorial is to clearly show that an agent is an LLM, a loop, and some tools.

We will build that loop in stages:

- `agent-00.py`: a plain terminal chatbot
- `agent-01.py`: the chatbot gets local file tools
- `agent-02.py`: the agent gets persistent memory

By the end, the agent can chat, inspect files, edit files, and remember facts between runs.

---

## 1. Setup

Create a project and install the dependencies:

```bash
mkdir build-an-agent
cd build-an-agent
uv venv
source .venv/bin/activate
```

> [!NOTE]
> We use uv for package management in this tutorial, but you can switch to `pip` or `conda` if necessary.

Create a `.env` file and add a token from Hugging Face. _(here are [instructions for creating a token](https://huggingface.co/docs/hub/en/security-tokens))_:

```bash
HF_TOKEN=your_huggingface_token_here
```

The examples use HuggingFace's OpenAI-compatible router:

```python
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
```

That lets us call an open model with the familiar `client.chat.completions.create(...)` API.

---

## 2. Stage 00: The Smallest Useful Loop

Start with `agent-00.py`.

This version is not an agent yet.  
Why not? Because of **agency**... the ability to do something.  

At this point, it's just a **chatbot**:

```python
def main():
  history = [
          {
              "role": "system",
              "content": (
                  "You are a helpful AI assistant."
                  "Offer all the help you reasonably can and decline when you cannot offer any help."
              )
          }
      ]

  while True:
      user = input("\033[96mYou\033[0m: ") # \033[96m<text>\033[0m colors the user's indicator text blue
      history.append({"role": "user", "content": user})

      completion = client.chat.completions.create(
          model="Qwen/Qwen2.5-7B-Instruct:cheapest",
          messages=history,
      )

      response = completion.choices[0].message.content.strip()
      print(f"\033[93mAgent\033[0m: {response}") # \033[93m<text>\033[0m colors the user's indicator text yellow
      history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
```

That is the heartbeat of every chat app:

1. Read user input.
2. Add it to the conversation.
3. Send the conversation to the model.
4. Print the model response.
5. Add the response back to the conversation.
6. Repeat.

The important detail is `history`. The model server is stateless. It does not remember your previous message unless you send the previous messages again. Our Python list is the agent's short-term memory, and also contains the system prompt, setting up the agent's default persona and context.

Run it:

```bash
uv run python agent-00.py
```

Try this:

```text
You: My name is Kwadwo. Give me one sentence about agents.
Agent: ...
You: What is my name?
Agent: ...
```

It should remember your name because `history` is sent on every request.

Still, this is only chat. It cannot **do** anything outside the conversation.

---

## 3. What Makes It an Agent?

For this tutorial, we will use a practical definition:

> An agent is an LLM in a loop with access to tools that can affect or inspect something outside the context window. Essentially, an LLM in a loop that can do stuff beyond just talking.

Tool use is simpler than it looks.

We tell the model:

```text
If you need to use a tool, reply only with a tool call like read_file("README.md").
```

Then our Python code:

1. Checks whether the model replied with a tool call.
2. Runs the matching Python function.
3. Sends the tool result back to the model.
4. Lets the model continue.

That is it. The model "asks" for a tool by printing text in a format we agreed on.

---

## 4. Stage 01: Local File Tools

`agent-01.py` adds three tools:

```python
TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "edit_file": edit_file,
}
```

Each tool is just a Python function which is registered in a dictionary:

```python
def read_file(path):
    """Return the text contents of a file at the given path."""
    file_path = Path(path)
    if not file_path.exists():
        return f"File not found: {path}"
    if file_path.is_dir():
        return f"{path} is a directory, not a file."
    return file_path.read_text()


def list_files(path="."):
    """Return a newline-separated listing of files and directories at path."""
    directory = Path(path)
    if not directory.exists():
        return f"Path not found: {path}"
    if not directory.is_dir():
        return f"{path} is not a directory."
    return "\n".join(sorted(item.name for item in directory.iterdir()))


def edit_file(path, old_str, new_str):
    """Replace the first occurrence of old_str with new_str in a file."""
    file_path = Path(path)
    if not file_path.exists():
        file_path.write_text(new_str)
        return f"Created file {path}"

    content = file_path.read_text()
    if old_str not in content:
        return "old_str not found in file"

    file_path.write_text(content.replace(old_str, new_str, 1))
    return "OK"
```

The system prompt tells the model how to call them:

```python
"You are an AI assistant with access to these tools: "
"list_files(path='.'), read_file(path), edit_file(path, old_str, new_str). "
"When you need a tool, reply with exactly one tool call and no other text. "
"After receiving a tool result, answer the user or call another tool if needed."
```

The parser detects a response like this:

```text
read_file("agent-00.py")
```

and turns it into:

```python
tool_name = "read_file"
args = ["agent-00.py"]
```

The parser uses Python's `ast` module:

```python
def parse_tool_call(text):
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None, []

    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return None, []

    args = [ast.literal_eval(arg) for arg in expression.args]
    args.extend(ast.literal_eval(keyword.value) for keyword in expression.keywords)
    name = expression.func.id
    return name, args
```

`ast` stands for abstract syntax tree. It lets us ask Python to parse the model's tool call as a real Python expression, then inspect the parsed structure.

>[!TIP]
><details>
><summary>Why use <code>ast</code> instead of other methods?</summary>
>
> There are a few ways we could parse tool calls:
>
> - Use regex or `text.split(",")`.
> - Ask the model to return JSON.
> - Use `eval(...)`.
> - Use `ast.parse(...)` plus `ast.literal_eval(...)`.
>
> Regex and comma-splitting look simple, but they break once arguments stop being simple. For example, this call has an empty string argument:
>
> ```python
> edit_file("mytxt", "", "hello")
> ```
>
> A naive parser can accidentally drop `""` and call `edit_file("mytxt", "hello")`, which crashes because `new_str` is missing. The `ast` parser preserves the actual arguments:
>
> ```python
> tool_name = "edit_file"
> args = ["mytxt", "", "hello"]
> ```
>
> JSON is a good option too, and many production agents use structured tool calls instead of text parsing. But for this stage, Python-style calls are easier to read in the terminal:
>
> ```python
> read_file("agent-00.py")
> ```
>
> That format is also easy to parse with `ast`.
>
> Do not use `eval(...)` here. The model output is untrusted text. `eval(...)` would execute that text as Python code. `ast.parse(...)` only parses the text into a tree, and `ast.literal_eval(...)` only extracts literal values like strings, numbers, lists, dictionaries, booleans, and `None`.
>
> That is why this tutorial chooses `ast`: it keeps the friendly `tool_name("arg")` syntax while avoiding the most obvious parser bugs and the security risk of `eval`.
>
> For more background, see the [official Python `ast` documentation](https://docs.python.org/3/library/ast.html) and this beginner-friendly guide to [`ast.parse` and `ast.literal_eval`](https://runebook.dev/en/docs/python/library/ast/ast.parse).

</details>

<br/>

Then the loop executes it:

```python
result = TOOLS[tool_name](*args)
```

The key change from `agent-00.py` is the inner while loop:

```python
while True:
    response = run_inference(history)
    print(f"\033[93mAgent\033[0m: {response}")

    tool_name, args = parse_tool_call(response)
    if tool_name not in TOOLS:
        history.append({"role": "assistant", "content": response})
        break

    result = TOOLS[tool_name](*args)
    print(f"\033[92mtool\033[0m: {tool_name}({', '.join(args)}) -> {result}")

    history.append({"role": "assistant", "content": response})
    history.append(
        {
            "role": "user",
            "content": f"Tool result for {tool_name}: {result}",
        }
    )
```

Why an inner loop? Because after a tool runs, we do not want to wait for the user. We want to send the result back to the model immediately so it can finish the job.

To see this in action, run:

```bash
uv run python agent-01.py
```

Try:

```text
You: What files are in this directory?
Agent: list_files(".")
tool: list_files(.) -> agent-00.py
agent-01.py
agent-tutorial.md
...
Agent: This directory contains ...
```

Now try:

```text
You: Read agent-00.py and explain the control flow.
```

At this point the model can inspect your project. It is no longer just answering from its training data.

You can also try:

```text
You: Read the files in the test-files folder and tell me what they say.
```

> [!WARNING]
> `edit_file` can write to your working directory. That is useful, but it is also the first dangerous tool in this tutorial.
>
> This version intentionally keeps the implementation small so you can see the moving pieces. A production agent should add guardrails:
> - Restrict file access to a project root.
> - Ask for confirmation before editing.
> - Show diffs before writing.
> - Log every tool call.
> - Avoid giving the model arbitrary shell access until you have a permission model.

---

## 5. Stage 02: Persistent Memory

`agent-00.py` already has memory, but only inside the current process. If you quit the program, `history` disappears.

`agent-02.py` adds durable memory with a JSON file:

```python
MEMORY_PATH = Path(".agent-memory.json")
```

The memory tools are:

```python
TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "edit_file": edit_file,
    "remember": remember,
    "recall": recall,
    "forget": forget,
}
```

The simplest memory tool is `remember`:

```python
def remember(key, value):
    memory = load_memory()
    memory[key] = value
    save_memory(memory)
    return f"Remembered {key}"
```

`recall` reads either one key or the whole memory file:

```python
def recall(key=""):
    memory = load_memory()
    if key:
        return memory.get(key, f"No memory found for {key}")
    if not memory:
        return "No memories stored yet."
    return json.dumps(memory, indent=2, sort_keys=True)
```

The system prompt is regenerated on each model call:

```python
def system_prompt():
    return (
        "You are an AI assistant with access to these tools: "
        "list_files(path='.'), read_file(path), edit_file(path, old_str, new_str), "
        "remember(key, value), recall(key=''), forget(key). "
        "When you need a tool, reply with exactly one tool call and no other text. "
        "Use memory for durable user preferences and facts that should survive restarts. "
        "Current memory:\n"
        f"{recall()}"
    )
```

That means the model can see stored memory even after restarting the program.

Run it:

```bash
uv run python agent-02.py
```

Try:

```text
You: Remember that my preferred Python package manager is uv.
Agent: remember("preferred_package_manager", "uv")
tool: remember(preferred_package_manager, uv) -> Remembered preferred_package_manager
Agent: I'll remember that your preferred Python package manager is uv.
```

Quit with `Ctrl+C`, run `agent-02.py` again, then ask:

```text
You: What package manager do I prefer?
```

The answer should come from `.agent-memory.json`.

---

## 6. Where This Goes Next

The first three stages give us the core architecture:

- `agent-00.py`: conversation loop
- `agent-01.py`: tool execution loop
- `agent-02.py`: persistent memory

The next natural stages are:

- `agent-03.py`: add a safer edit flow that previews diffs before writing
- `agent-04.py`: add shell commands with an allowlist and confirmation step
- `agent-05.py`: replace text-based tool calls with provider-native tool calling
- `agent-06.py`: add planning, task state, and a maximum tool-call budget

The pattern is the same: give the model a capability, detect when it wants to use it, execute it in ordinary Python, then return the result.

The quality depends on **the loop, the tool design, and the constraints** you put around what the agent is allowed to do.

>[!TIP] BIG IDEA
>What we have built here, while basic, is already more powerful than you think!  
>
> Try `"Create a prime_number_generator.py"` and you will see that our little agent can already successfully generate code, and by extension, shell commands. This means it has the potential to be extended to control a computer and create any new functionality it needs!

If you prefer videos, here's one by Geoffrey Huntley, [building an agent live on stage!](https://www.youtube.com/watch?v=OR3zdu9T_as)

What will you build next?! 😃

---

**Attribution:** This tutorial is based on the structure and ideas in [How to Build an Agent](https://ampcode.com/notes/how-to-build-an-agent) by Thorsten Ball. This Python version uses HuggingFace's OpenAI-compatible Inference API and staged examples instead of the original Go/Anthropic implementation.
