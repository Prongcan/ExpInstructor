# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com")

def chat_deepseek(prompt):
    response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
            {"role": "system", "content": "You need to engage in deep thinking."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    #print(f"Model is: {response.model}")
    #print(f"Output is: {response.choices[0].message.content}")
    print(response.choices[0].message.reasoning_content)
    return response.choices[0].message.content
