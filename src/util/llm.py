
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import streamlit as st
from config import LLM_MODEL, EMBEDDING_MODEL, top_K, sqlite_table
from langchain_ollama import OllamaEmbeddings
import sqlite3
import re
from tabulate import tabulate
from langgraph.types import RetryPolicy

class GraphState(TypedDict):
    question: str
    intent: Optional[str]
    result: Optional[Any]
    last_errpr: str

router_prompt = PromptTemplate.from_template("""
    Classify the following question into one of the categories below:

    general: General or high-level questions that are informational in nature and not directly tied to fraud documents or historical fraud data.

    fraud_doc: Questions specifically related to fraud documentation, including explanations of fraud types, fraud techniques, prevention methods, impacts of fraud, or conceptual discussions about credit card fraud (e.g., card-related frauds, merchant frauds, internet frauds, fraud prevention technologies).

    fraud_history: Questions related to historical fraud data, fraud records, trends, statistics, or analysis that would typically require querying databases or using SQL.

    Answer using only one of the labels above.

    Question:
    {question}
    """)

def init_llm(url):
    return ChatOllama(model=LLM_MODEL, base_url=url), OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=url)

def router_node(state: GraphState):
    response = st.session_state.llm.invoke(
        router_prompt.format(question=state["question"])
    )

    return {
        "intent": response.content.strip()
    }
    
    
def general_llm_node(state: GraphState):
    return {
        "result": {
            "stream": st.session_state.llm.stream(state["question"]),
            "references": None
        }
    }
    
def fraud_doc_node(state: GraphState):
    
    query_prompt = f"""
    You are a rephrasing agent.
    Rewrite the user's input into a clear, well-structured, and semantically rich query that maximizes similarity matching in a vector database.

    Preserve the original intent and meaning, but improve clarity and wording.
    Do not add new information or change the intent.

    User input:
    {state["question"]}
    """
    
    query = st.session_state.llm.invoke(query_prompt).content
    
    docs = st.session_state.vectordb.similarity_search(query, k=top_K)

    context = "\n\n".join([
    f"--- Result {i+1} ---\n\n{doc.page_content}"
    for i, doc in enumerate(docs)])

    prompt = f"""
    Based on the following documents, answer the question and include a brief reference.

    Documents:
    {context}

    Question:
    {state["question"]}

    Additional instruction:
    Provide a concise summary with a maximum length of 200 words.
    """

    return {
        "result": {
            "stream": st.session_state.llm.stream(prompt),
            "references": f"{context} \n\n Summary : "
        }
    }

def extract_sql(text):
    pattern = r"```sql\s*(.*?)\s*```|:"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def fraud_history_node(state: GraphState):
    conn = sqlite3.connect(st.session_state.db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (sqlite_table,)
        )

        row = cursor.fetchone()
        if not row or not row[0]:
            raise ValueError(f"Table '{sqlite_table}' does not exist")

        ddl = row[0]

        error_feedback = ""
        if state.get("last_error"):
            error_feedback = f"""
            Previous SQL error:
            {state['last_error']}

            Fix the SQL accordingly.
            """

        sql_prompt = f"""
        You are an expert SQL generator.

        Rules:
        - SQLite dialect only
        - SELECT queries only
        - Use only schema provided
        - Always LIMIT 100
        - Return ONLY SQL
        - get only relevant column

        DDL:
        {ddl}

        User request:
        {state["question"]}

        {error_feedback}
        """

        sql_query = st.session_state.llm.invoke(sql_prompt).content
        sql = extract_sql(sql_query).strip()
        
        print(sql)

        if not sql.lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed")

        if "limit" not in sql.lower():
            sql += " LIMIT 100"

        try:
            cursor.execute(sql)
        except Exception as e:
            raise ValueError(f"SQL execution error: {str(e)}")

        if cursor.description is None:
            raise ValueError("Query returned no columns")

        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(100)

        sql_result = tabulate(rows, headers=columns, tablefmt="psql")

        summary_prompt = f"""
        Based on the following information:

        {sql_result}

        Provide a concise summary (max 200 words).
        """

        return {
            "result": {
                "references": f"{sql_result}\n\nSummary:",
                "stream": st.session_state.llm.stream(summary_prompt),
            }
        }

    except Exception as e:
        state["last_error"] = str(e)
        print(str(e))
        raise 

    finally:
        conn.close()



graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("general", general_llm_node)
graph.add_node("fraud_doc", fraud_doc_node)
graph.add_node("fraud_history", fraud_history_node,    
        retry=RetryPolicy(
        max_attempts=3,
    ))

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["intent"],
    {
        "general": "general",
        "fraud_doc": "fraud_doc",
        "fraud_history": "fraud_history"
    }
)

graph.add_edge("general", END)
graph.add_edge("fraud_doc", END)
graph.add_edge("fraud_history", END)

app = graph.compile()