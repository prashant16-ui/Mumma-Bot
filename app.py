import streamlit as st
from openai import OpenAI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from retriever import get_vectorstore 
from prompt import system_prompt
from logger import log_to_google_sheets
import uuid

# Page config
st.set_page_config(page_title="Maa-Saathi", page_icon="🤰")

st.title("🤰 Maa-Saathi")
st.caption("Your AI companion for pregnancy & motherhood")

# Load API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load vectorstore
vectorstore = get_vectorstore()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Add session ID (IMPORTANT)
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

    # 🔍 Retrieval + Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤰"):

            try:
                # Retrieve docs
                docs = vectorstore.similarity_search(user_input, k=3)

                # Safe context handling
                if docs:
                    context = "\n\n".join([doc.page_content for doc in docs])
                else:
                    context = "No relevant context found."

                # Limit context size
                context = context[:3000]

                # Format prompt
                final_prompt = system_prompt.format(context=context)

                # Generate response
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": final_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.5
                )

                reply = response.choices[0].message.content

            except Exception as e:
                reply = "⚠️ Sorry, I'm having trouble right now. Please try again."
                docs = []  # ensure docs exists

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

    # ✅ Prepare logging data
    sources = [doc.metadata.get("source", "Unknown") for doc in docs] if docs else []

    # ✅ Log to Google Sheets (SAFE)
    try:
        log_to_google_sheets(
            user_input,
            reply,
            sources,
            st.session_state.session_id
        )
    except Exception as e:
        print("Logging error:", e)

    # Refresh UI
    st.rerun()