from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

# モデルの設定
model = ChatOllama(model="phi4-mini")

# プロンプトテンプレートの設定
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="あなたはクラウド技術に関するプロフェッショナルなエンジニアとして回答するAIアシスタントです。"),
        MessagesPlaceholder("history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# 履歴の設定
history = [
    HumanMessage(content="こんにちは、クラウドエンジニアリングについて教えてください。"),
    AIMessage(content="こんにちは！クラウドエンジニアリングは、クラウドインフラストラクチャの設計、構築、運用を行う技術分野です。具体的には、AWS、Azure、GCPなどのクラウドサービスを活用して、スケーラブルで信頼性の高いシステムを構築します。"),
    HumanMessage(content="クラウドアーキテクチャの基本的な構成要素を教えてください。"),
    AIMessage(content="クラウドアーキテクチャの基本的な構成要素には、コンピューティング（例: 仮想マシンやコンテナ）、ストレージ（例: オブジェクトストレージやブロックストレージ）、ネットワーク（例: VPCやロードバランサー）が含まれます。これらを組み合わせて、効率的なシステムを構築します。")
]

# チェーンの作成
chain = prompt_template | model

# チェーンを使って質問を処理
ai_msg = chain.invoke(
    {
        "history": history,
        "question": "クラウドセキュリティのベストプラクティスを教えてください。"
    }
)

# contentのみ抽出して表示
print(ai_msg.content)