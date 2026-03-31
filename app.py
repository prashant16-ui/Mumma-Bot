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

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Maa-Saathi", page_icon="🤰")

st.title("🤰 Maa-Saathi")
st.caption("Your AI companion for pregnancy & motherhood")

# ------------------ CHECK SECRETS ------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OpenAI API key not found. Please configure secrets.")
    st.stop()

# ------------------ INIT CLIENT ------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------ LOAD VECTORSTORE ------------------
try:
    vectorstore = get_vectorstore()
except Exception as e:
    st.error("❌ Failed to load knowledge base. Please try again later.")
    st.stop()

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ USER INPUT ------------------
user_input = st.chat_input("Ask Maa-Saathi anything...")

if user_input:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------ ASSISTANT RESPONSE ------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤰"):

            try:
                # 🔍 Retrieve relevant docs
                docs = vectorstore.similarity_search(user_input, k=3)

                # Build context safely
                context = "\n\n".join([
                    doc.page_content for doc in docs if doc.page_content
                ])
                context = context[:1500]

                # Format prompt
                final_prompt = system_prompt.format(context=context)

                # 🤖 Generate response
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
                # Handle quota issue separately
                if "insufficient_quota" in str(e):
                    reply = "⚠️ Service temporarily unavailable. Please try again later."
                else:
                    st.error("⚠️ Something went wrong.")
                    st.text(traceback.format_exc())
                    reply = "⚠️ Backend error"
                docs = []

        # Show response
        st.markdown(reply)

        # ------------------ SOURCES ------------------
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

    # ------------------ LOGGING ------------------
    try:
        sources = [doc.metadata.get("source", "Unknown") for doc in docs] if docs else []

        log_to_google_sheets(
            user_input,
            reply,
            sources,
            st.session_state.session_id
        )

    except Exception as e:
        pass  # silent fail (don't break UX)

    # ------------------ REFRESH ------------------
    if reply not in ["⚠️ Backend error"]:
        st.rerun()