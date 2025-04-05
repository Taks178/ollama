from typing_extensions import TypedDict
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from pprint import pprint

# GraphStateの定義
class GraphState(TypedDict):
    user_question: str  # ユーザーの質問
    generated_answer: str  # 生成された回答
    reference_document: str  # 取得した参考情報

# モデル設定
ollama_model_name = "phi4-mini"

# プロンプトテンプレートの定義
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("{system_message}"),
    HumanMessagePromptTemplate.from_template("{user_message}")
])

# ユーザーの質問に基づいて、使用するツールを決定するchainの設定
tool_selection_system_message = """あなたはユーザーからの質問に対し、Tool使用の要否を決定する役割を持っています。
質問内容について、"天気"に関する質問の場合は"weather"、それ以外は"no_tool"と答えてください。
出力は必ず次のようなJSON形式にしてください:
{
    "tool": "weather" または "no_tool"
}
"""
tool_selection_prompt = chat_prompt_template.partial(system_message=tool_selection_system_message)
tool_selection_llm = ChatOllama(model=ollama_model_name, format="json")
tool_selection_chain = tool_selection_prompt | tool_selection_llm | JsonOutputParser()

# 回答生成のためのchainの設定
answer_generation_system_message = """あなたはユーザーからの質問に対し、回答する役割を持っています。
もし質問と一緒に関連情報が与えられている場合は必ず関連情報を見て回答を生成してください。
"""
answer_generation_prompt = chat_prompt_template.partial(system_message=answer_generation_system_message)
answer_generation_llm = ChatOllama(model=ollama_model_name)
answer_generation_chain = answer_generation_prompt | answer_generation_llm | StrOutputParser()

# 各ノードの関数定義
def fetch_weather_info(state: GraphState):
    user_question = state.get("user_question", "")
    # 本来はここで天気APIから情報を取得する
    reference_document = """埼玉の天気は必ず晴れです。埼玉では雨は降りません。"""
    return {"user_question": user_question, "reference_document": reference_document}

def generate_answer(state: GraphState):
    relevant_document = state.get("reference_document", "なし")
    user_question = state.get("user_question", "")
    user_message = (
        f"関連情報:{relevant_document}\n"
        f"質問:{user_question}"
    )
    generated_answer = answer_generation_chain.invoke({"user_message": user_message})
    return {"generated_answer": generated_answer}

# toolを選択するの関数定義
def decide_tool_selection(state: GraphState):
    user_question = state.get("user_question", None)
    selection_result = tool_selection_chain.invoke({"user_message": user_question})
    tool_selection = selection_result.get("tool", "no_tool")
    return tool_selection

# ワークフローの構築
workflow_graph = StateGraph(GraphState)

# ノードの追加
workflow_graph.add_node("fetch_weather_info", fetch_weather_info)
workflow_graph.add_node("generate_answer", generate_answer)

# ノード間の接続
workflow_graph.add_edge("fetch_weather_info", "generate_answer")
workflow_graph.add_edge("generate_answer", END)

# 条件分岐による接続先ノードのスイッチング
workflow_graph.add_conditional_edges(
    START,
    decide_tool_selection, # この戻り値に基づいて接続先のnodeを決定
    {
        "weather"  : "fetch_weather_info",
        "no_tool"  : "generate_answer"
    },
)

# ワークフローのコンパイル
agent = workflow_graph.compile()

# 入力をもとにエージェントを実行
inputs = {
    "user_question": "明日の東京の天気は？"
    # "user_question": "明日の埼玉の天気は？"
}
for output in agent.stream(inputs):
    for key, value in output.items():
        pprint(f"{key}: {value}")
    print("===")
