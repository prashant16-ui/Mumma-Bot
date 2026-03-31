import streamlit as st
from openai import OpenAI
import sys
import os
import uuid
import traceback

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from retriever import get_vectorstore 
from prompt import system_prompt
from logger import log_to_google_sheets

# Page config
st.set_page_config(page_title="Maa-Saathi", page_icon="🤰")

st.title("🤰 Maa-Saathi")
st.caption("Your AI companion for pregnancy & motherhood")

# ✅ Check secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY missing in Streamlit secrets")
    st.stop()

# Load OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load vectorstore safely
try:
    vectorstore = get_vectorstore()
    st.write("✅ Vectorstore loaded")
except Exception as e:
    st.error(f"❌ Vectorstore loading failed: {e}")
    st.stop()

# Debug secrets (safe keys only)
st.write("🔑 Secrets loaded:", list(st.secrets.keys()))

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask Maa-Saathi anything...")

if user_input:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤰"):

            try:
                # 🔍 Retrieve docs
                docs = vectorstore.similarity_search(user_input, k=3)

                # Context
                context = "\n\n".join([
                    str(doc.page_content) for doc in docs if doc.page_content
                ])
                context = context[:1500]

                # Prompt
                final_prompt = system_prompt.format(context=context)

                # 🤖 LLM
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": final_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.5
                )

                reply = response.choices[0].message.content

            except Exception as e:
                st.error("🔥 FULL ERROR BELOW:")
                st.code(traceback.format_exc())
                reply = "⚠️ Backend crashed"
                docs = []

        # Show response
        st.markdown(reply)

        # 📚 Sources
        if docs:
            with st.expander("📚 Sources"):
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown")
                    st.write(f"**Source {i+1}:** {source}")

    # Store assistant reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    # Prepare logging data
    sources = [doc.metadata.get("source", "Unknown") for doc in docs] if docs else []

    # 📊 Logging (safe)
    try:
        log_to_google_sheets(
            user_input,
            reply,
            sources,
            st.session_state.session_id
        )
    except Exception as e:
        st.warning(f"⚠️ Logging failed: {e}")

    # Refresh UI
    st.rerun()
