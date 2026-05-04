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
