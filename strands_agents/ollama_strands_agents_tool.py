from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import current_time, python_repl

@tool
def letter_counter(word: str, letter: str) -> int:
    return word.lower().count(letter.lower())

ollama_model = OllamaModel(
    host="http://localhost:11434",
    # model_id="phi4-mini"
    model_id="qwen3" 
)

agent = Agent(model=ollama_model, tools=[letter_counter])

message = """
この文章の中に「1」はいくつある？
"""

agent(message)
