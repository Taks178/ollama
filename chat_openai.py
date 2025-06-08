from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="phi4-mini",
  messages=[
    {"role": "system", "content": "あなたはIT技術について詳しい、優秀なアシスタントです。"},
    {"role": "user", "content": "AWSとは何ですか？"}
  ]
)
print(response.choices[0].message.content)