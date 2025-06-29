import boto3
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands.models import BedrockModel
from strands_tools import current_time, python_repl, use_aws

@tool
def letter_counter(word: str, letter: str) -> int:
    return word.lower().count(letter.lower())

# Ollamaモデルの例
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini" 
)

agent = Agent(model=ollama_model, tools=[letter_counter])

message = """
以下の文章の中に「1」はいくつある？toolのletter_counterを使って答えてください。
対象文章：「この文章には1が1つふくまれています。」
"""

agent(message)
