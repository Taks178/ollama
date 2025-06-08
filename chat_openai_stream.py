from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

stream = client.chat.completions.create(
    model="qwen3",
    # model="phi4-mini",
    messages=[
        {"role": "system", "content": "あなたはIT技術について詳しい、優秀なアシスタントです。"},
        {"role": "user", "content": "AWSとは何ですか？"},
    ],
    stream=True,
)

for event in stream:
    print(event.choices[0].delta.content, end="")