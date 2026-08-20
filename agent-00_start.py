import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

MODEL = "" # Select a model from https://huggingface.co/inference/models

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

def main():
    print("Chat with your agent (Ctrl+C to quit)")
    history = [
        {
            # Fill out your model's personality
            "role": "system",
            "content": (
                
            )
        }
    ]
    while True:
        user = input("\033[96mYou\033[0m: ")
        history.append({"role": "user", "content": user})
        messages = []
        for turn in history:
            #add turn logic
            break
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        response = completion.choices[0].message.content.strip()
        print(f"\033[93mAgent\033[0m: {response}")

        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
