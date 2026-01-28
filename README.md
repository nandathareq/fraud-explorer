# fraud-explorer

📚 Fraud Explorer
An interactive Streamlit web app that lets you search, ask, and discuss fraud data and document using an LLM (via LangChain + Ollama). The app creates a collaborative "Fraud Explorer" interface where you can ask questions, retrieve Fraud History, and explore fraud documentation.

🚀 Features
🔍 Search fraud incident history by text in english
🧠 Semantic search across stored fraud articles using chromaDB
💬 Chat interface to discuss

fraud-explorer/
│── src/
│   ├── main.py               # Main entry (Streamlit app)
│   ├── config.py             # App configuration
│   └── util/
│       ├── data.py           # Data ingestion pipeline
│       └── llm.py            # LLM wrapper + tool orchestration
│
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
└── .gitignore                # Git ignore rules



⚙️ Installation
Clone the repository:

git clone https://github.com/nandathareq/fraud-explorer.git
cd fraud-explorer
Create & activate a virtual environment:

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
Install dependencies:

pip install -r requirements.txt

▶️ Usage
Run the Streamlit app:

streamlit run src/main.py
Then open in your browser: 👉 http://localhost:8501

🛠️ Configuration
You can adjust settings in config.py and make sure it match with LLM Engine Collab:

LLM model (default: qwen2.5:7b)
Embedding model (default: nomic-embed-text)
Vector store search parameters


📌 Example Workflow
Open App
run LLM Engine in Collab
paste ngrok tunnel into popup
Start a chat in the Chat tab

Ask about a topic (e.g., "find 10 most recent fraud incident")

The agent may call tools automatically to:

Search for a relevant document
Search for incident history