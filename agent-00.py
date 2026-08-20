import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

def main():
    print("Chat with your agent (Ctrl+C to quit)")
    history = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Offer all the help you reasonably can and decline when you cannot offer any help."
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
            model="Qwen/Qwen3-8B",
            messages=messages,
        )
        response = completion.choices[0].message.content.strip()
        print(f"\033[93mAgent\033[0m: {response}")

        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
