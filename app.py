import streamlit as st

from src.crew import run_mednotes
from src.memory.chat_store import ChatStore


st.set_page_config(
    page_title="MedNotes RAG",
    page_icon="🩺",
    layout="wide"
)


store = ChatStore()


if "chat_id" not in st.session_state:
    st.session_state.chat_id = store.create_chat()


st.title("🩺 MedNotes RAG")
st.caption("Clinical Document QA Assistant")


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Chats")


if st.sidebar.button("➕ New Chat"):

    st.session_state.chat_id = store.create_chat()

    st.rerun()


for chat_id, created_at in store.get_all_chats():

    if st.sidebar.button(
        f"Chat {chat_id}",
        key=f"chat_{chat_id}"
    ):

        st.session_state.chat_id = chat_id

        st.rerun()


# -----------------------------
# Chat History
# -----------------------------

history = store.get_history(
    st.session_state.chat_id
)


for message in history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# -----------------------------
# User Input
# -----------------------------

question = st.chat_input(
    "Ask a medical question..."
)


if question:

    store.add_message(
        st.session_state.chat_id,
        "user",
        question
    )


    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching clinical notes..."
        ):

            answer = run_mednotes(
                question
            )

            st.markdown(
                answer
            )


    store.add_message(
        st.session_state.chat_id,
        "assistant",
        answer
    )


    st.rerun()