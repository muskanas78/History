import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
conversation = []
history_file = "chat_history.txt"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def save_to_history(user_msg, bot_reply):
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"You: {user_msg}\nLLaMA: {bot_reply}\n\n")

def send_message(user_text, image_url=None):
    content = [{"type": "text", "text": user_text}]
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    conversation.append({"role": "user", "content": content})

    completion = client.chat.completions.create(
        model="qwen/qwen2.5-vl-32b-instruct:free",
        messages=conversation,
        extra_headers={
            "HTTP-Referer": "https://github.com/muskanas78",
            "X-Title": "AI Chatbot",
        },
    )

    reply = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    return reply

print("LLaMA Chat (Multimodal) - Type 'exit' to quit")
print("-" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == 'exit':
        break

    # ask if image is needed
    add_image = input("Do you want to include an image? (yes/no): ").strip().lower()
    image_url = None
    if add_image == "yes":
        image_url = input("Paste the image URL: ").strip()

    response = send_message(user_input, image_url)
    print("LLaMA:", response)
    save_to_history(user_input, response)
