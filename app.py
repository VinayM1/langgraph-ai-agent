import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import agent

import sys, os
sys.path.append(os.path.dirname(__file__))
# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide"
)

# ---------- STYLES ----------
st.markdown("""
<style>
.chat-bubble {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user {
    background-color: #1f77b4;
    color: white;
}
.assistant {
    background-color: #2c2f33;
    color: white;
}
.tool {
    background-color: #444;
    color: #ddd;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("⚙️ Settings")

    show_tools = st.toggle("Show Tool Calls", True)
    show_debug = st.toggle("Debug Mode", False)

    st.markdown("---")
    st.markdown("### 🧠 About")
    st.write("LangGraph + RAG + Tools powered agent")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# ---------- STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- TITLE ----------
st.title("🤖 AI Agent")
st.caption("LangGraph • RAG • Tools • Streamlit UI")

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

    elif msg["role"] == "tool" and show_tools:
        with st.expander("🔧 Tool Output"):
            st.code(msg["content"])

# ---------- INPUT ----------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.spinner("Thinking..."):
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)],
            "llm_calls": 0
        })

    final_answer = ""
    tool_logs = []

    # Process messages
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_logs.append(
                        f"Tool: {tc['name']}\nArgs: {tc['args']}"
                    )
            final_answer = msg.content

        elif isinstance(msg, ToolMessage):
            tool_logs.append(msg.content)

    # Display assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()

        # fake streaming effect
        streamed = ""
        for char in final_answer:
            streamed += char
            placeholder.markdown(streamed)
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })

    # Show tools
    if show_tools and tool_logs:
        with st.expander("🔧 Tool Execution Details"):
            for log in tool_logs:
                st.code(log)

    # Debug panel
    if show_debug:
        with st.expander("🐞 Debug Data"):
            st.json(result)
