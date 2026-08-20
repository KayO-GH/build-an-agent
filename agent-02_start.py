import ast
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "Qwen/Qwen3-8B"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MEMORY_PATH = Path(".agent-memory.json")


# add logic
def load_memory():
    return


def save_memory(memory):
    return


def remember(key, value):
    """Store a durable memory value under the given key."""
    return


def recall(key=""):
    """Return one remembered value, or all memories when no key is given."""
    return 


def forget(key):
    """Remove a durable memory value by key."""
    return


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


TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "edit_file": edit_file,
    "remember": remember,
    "recall": recall,
    "forget": forget,
}


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


def run_inference(history):
    messages = [{"role": "system", "content": system_prompt()}] + history
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content.strip()


def main():
    print("Chat with your agent (Ctrl+C to quit)")
    history = []

    while True:
        user = input("\033[96mYou\033[0m: ")
        history.append({"role": "user", "content": user})

        while True:
            response = run_inference(history)
            print(f"\033[93mAgent\033[0m: {response}")

            tool_name, args = parse_tool_call(response)
            if tool_name not in TOOLS:
                history.append({"role": "assistant", "content": response})
                break

            try:
                result = TOOLS[tool_name](*args)
            except TypeError as error:
                result = f"Tool call error: {error}"
            print(f"\033[92mtool\033[0m: {tool_name}({', '.join(map(str, args))}) -> {result}")

            history.append({"role": "assistant", "content": response})
            history.append(
                {
                    "role": "user",
                    "content": f"Tool result for {tool_name}: {result}",
                }
            )


if __name__ == "__main__":
    main()
