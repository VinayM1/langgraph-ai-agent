import operator
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Annotated, Literal
from typing_extensions import TypedDict

import streamlit as st

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph



load_dotenv()


if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]



# STATE
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int



# TOOLS
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides a by b.

    Args:
        a: First int
        b: Second int
    """
    if b == 0:
        return 0.0   
    return a / b


@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for AI/ML, LangGraph, RAG, embeddings, etc.

    Args:
        query: The search query describing what information you need
    """

    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")


    if not os.path.exists(index_path):
        from ingest import main
        main()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    return "\n\n---\n\n".join(doc.page_content for doc in docs)


tools = [add, multiply, divide, search_docs]
tools_by_name = {t.name: t for t in tools}



# MODEL
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

model_with_tools = model.bind_tools(tools)


# GRAPH NODES
def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or return a final answer."""
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant that can perform arithmetic "
                    "and answer questions about AI/ML concepts. "
                    "Use the search_docs tool when needed. "
                    "Use math tools for calculations."
                )
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState) -> dict:
    """Execute every tool call requested by the LLM."""
    results = []

    for tool_call in state["messages"][-1].tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"

    return END


# BUILD GRAPH
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
