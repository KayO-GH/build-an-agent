import ast
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


def read_file(path):
    return


def list_files(path="."):
    """Return a newline-separated listing of files and directories at path."""
    return


def edit_file(path, old_str, new_str):
    """Replace the first occurrence of old_str with new_str in a file."""
    return


TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "edit_file": edit_file,
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


def run_inference(history):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=history,
    )
    return completion.choices[0].message.content.strip()


def main():
    print("Chat with your agent (Ctrl+C to quit)")
    history = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant with access to these tools: "
                "list_files(path='.'), read_file(path), edit_file(path, old_str, new_str). "
                "When you need a tool, reply with exactly one tool call and no other text. "
                "After receiving a tool result, answer the user or call another tool if needed."
            ),
        }
    ]

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
