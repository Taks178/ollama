from ollama import chat

stream = chat(
    model='phi4-mini',
    messages=[{
        'role': 'user', 
        'content': '空はなぜ青いのか？'
    }],
    stream=True,
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)