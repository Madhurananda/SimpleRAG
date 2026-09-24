# """Unified Streamlit app for all RAG assistants (HR, Interspeech, Bible).

# Run with:
#     streamlit run app_unified.py --server.address 0.0.0.0 --server.port 8501
# """

# import streamlit as st

# # Import HR pipeline
# from hr_assistant.pipeline import ask as ask_hr, build_hr_assistant

# # Import Interspeech pipeline
# from interspeech.pipeline import ask as ask_interspeech, build_interspeech_assistant

# # ============================================================
# # PAGE CONFIG
# # ============================================================
# st.set_page_config(
#     page_title="RAG Assistant Hub",
#     page_icon="🤖",
#     layout="wide"
# )

# st.title("🤖 Multi‑RAG Assistant Hub")
# st.caption("Choose a tab below to chat with a specific assistant.")

# # ============================================================
# # SESSION STATE INITIALIZATION
# # ============================================================
# if "hr_messages" not in st.session_state:
#     st.session_state.hr_messages = []
# if "interspeech_messages" not in st.session_state:
#     st.session_state.interspeech_messages = []
# if "bible_messages" not in st.session_state:
#     st.session_state.bible_messages = []

# # ============================================================
# # CACHED AGENT BUILDERS
# # ============================================================
# @st.cache_resource(show_spinner="Building HR Assistant...")
# def get_hr_agent():
#     """Build and cache the HR assistant agent."""
#     return build_hr_assistant()

# @st.cache_resource(show_spinner="Building Interspeech Assistant...")
# def get_interspeech_agent(_years, _filter_dementia, _max_papers):
#     """Build and cache the Interspeech assistant agent."""
#     return build_interspeech_assistant(
#         years=_years,
#         filter_dementia=_filter_dementia,
#         max_papers=_max_papers if _max_papers > 0 else None,
#         rebuild=False
#     )

# def build_bible_agent():
#     """Placeholder for Bible assistant."""
#     # TODO: Implement Bible pipeline
#     return None

# # ============================================================
# # TABS
# # ============================================================
# tab1, tab2, tab3 = st.tabs(["🏢 HR Policy", "🎤 Interspeech", "📖 Bible"])

# # ------------------------------------------------------------
# # TAB 1: HR ASSISTANT
# # ------------------------------------------------------------
# with tab1:
#     st.header("HR Policy Assistant")
#     st.caption("Ask questions about your company's HR policies.")

#     # Build the agent (cached)
#     hr_agent = get_hr_agent()

#     # Show chat history
#     for message in st.session_state.hr_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     question = st.chat_input("Ask a question about HR policy...", key="hr_input")
#     if question:
#         st.session_state.hr_messages.append({"role": "user", "content": question})
#         with st.chat_message("user"):
#             st.markdown(question)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 answer = ask_hr(hr_agent, question)
#             st.markdown(answer)
#         st.session_state.hr_messages.append({"role": "assistant", "content": answer})

# # ------------------------------------------------------------
# # TAB 2: INTERSPEECH ASSISTANT
# # ------------------------------------------------------------
# with tab2:
#     st.header("Interspeech Research Assistant")
#     st.caption("Ask questions about research presented at Interspeech conferences.")

#     # Settings sidebar inside the tab
#     with st.expander("⚙️ Settings", expanded=False):
#         col1, col2 = st.columns(2)
#         with col1:
#             years = st.multiselect(
#                 "Years",
#                 options=[2020, 2021, 2022, 2023, 2024, 2025],
#                 default=[2025],
#                 key="is_years"
#             )
#             filter_dementia = st.checkbox("🧠 Only dementia-related papers", value=True, key="is_dementia")
#         with col2:
#             max_papers = st.number_input(
#                 "Max papers per year",
#                 min_value=0,
#                 max_value=500,
#                 value=20,
#                 help="0 = load all papers",
#                 key="is_max_papers"
#             )
#             if st.button("🔄 Rebuild Index", key="is_rebuild"):
#                 st.cache_resource.clear()
#                 st.rerun()

#     # Build the agent (cached)
#     is_agent = get_interspeech_agent(years, filter_dementia, max_papers)

#     # Show chat history
#     for message in st.session_state.interspeech_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     question = st.chat_input("Ask a question about Interspeech research...", key="is_input")
#     if question:
#         st.session_state.interspeech_messages.append({"role": "user", "content": question})
#         with st.chat_message("user"):
#             st.markdown(question)

#         with st.chat_message("assistant"):
#             with st.spinner("Searching papers..."):
#                 answer = ask_interspeech(is_agent, question)
#             st.markdown(answer)
#         st.session_state.interspeech_messages.append({"role": "assistant", "content": answer})

# # ------------------------------------------------------------
# # TAB 3: BIBLE ASSISTANT (PLACEHOLDER)
# # ------------------------------------------------------------
# with tab3:
#     st.header("Bible Study Assistant")
#     st.caption("Coming soon! This will allow you to ask questions about the Bible.")

#     st.info("📖 The Bible assistant is not yet implemented. Check back later!")

#     # Placeholder chat interface
#     for message in st.session_state.bible_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     question = st.chat_input("Ask a question about the Bible... (placeholder)", key="bible_input")
#     if question:
#         st.session_state.bible_messages.append({"role": "user", "content": question})
#         with st.chat_message("user"):
#             st.markdown(question)
#         with st.chat_message("assistant"):
#             st.markdown("⚠️ This assistant is not yet implemented. Please select another tab.")
#         st.session_state.bible_messages.append({"role": "assistant", "content": "⚠️ This assistant is not yet implemented."})





"""Unified Streamlit app for all RAG assistants (HR, Interspeech, Bible).

Run with:
    streamlit run app_unified.py --server.address 0.0.0.0 --server.port 8501
"""

import streamlit as st

# Import HR pipeline
from assistants.hr.pipeline import ask as ask_hr, build_hr_assistant

# Import Interspeech pipeline
from interspeech.pipeline import ask as ask_interspeech, build_interspeech_assistant

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RAG Assistant Hub",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi‑RAG Assistant Hub")
st.caption("Choose a tab below to chat with a specific assistant.")

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "hr_messages" not in st.session_state:
    st.session_state.hr_messages = []
if "interspeech_messages" not in st.session_state:
    st.session_state.interspeech_messages = []
if "bible_messages" not in st.session_state:
    st.session_state.bible_messages = []

# ============================================================
# CACHED AGENT BUILDERS
# ============================================================
@st.cache_resource(show_spinner="Building HR Assistant...")
def get_hr_agent():
    """Build and cache the HR assistant agent."""
    return build_hr_assistant()

@st.cache_resource(show_spinner="Building Interspeech Assistant...")
def get_interspeech_agent(_years, _filter_dementia, _max_papers):
    """Build and cache the Interspeech assistant agent."""
    return build_interspeech_assistant(
        years=_years,
        filter_dementia=_filter_dementia,
        max_papers=_max_papers if _max_papers > 0 else None,
        rebuild=False
    )

# ============================================================
# REUSABLE CHAT INTERFACE
# ============================================================
def chat_interface(agent, messages, ask_func, input_key, placeholder):
    """Reusable chat interface with input at the bottom."""
    # Use a container to push the input to the bottom
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input (always at the bottom of the tab)
    question = st.chat_input(placeholder, key=input_key)
    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_func(agent, question)
            st.markdown(answer)
        messages.append({"role": "assistant", "content": answer})
        st.rerun()  # Force rerun to show the new message

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🏢 HR Policy", "🎤 Interspeech", "📖 Bible"])

# ------------------------------------------------------------
# TAB 1: HR ASSISTANT
# ------------------------------------------------------------
with tab1:
    st.header("HR Policy Assistant")
    st.caption("Ask questions about your company's HR policies.")

    hr_agent = get_hr_agent()
    chat_interface(
        agent=hr_agent,
        messages=st.session_state.hr_messages,
        ask_func=ask_hr,
        input_key="hr_input",
        placeholder="Ask a question about HR policy..."
    )

# ------------------------------------------------------------
# TAB 2: INTERSPEECH ASSISTANT
# ------------------------------------------------------------
with tab2:
    st.header("Interspeech Research Assistant")
    st.caption("Ask questions about research presented at Interspeech conferences.")

    # Settings expander
    with st.expander("⚙️ Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            years = st.multiselect(
                "Years",
                options=[2020, 2021, 2022, 2023, 2024, 2025],
                default=[2025],
                key="is_years"
            )
            filter_dementia = st.checkbox("🧠 Only dementia-related papers", value=True, key="is_dementia")
        with col2:
            max_papers = st.number_input(
                "Max papers per year",
                min_value=0,
                max_value=500,
                value=20,
                help="0 = load all papers",
                key="is_max_papers"
            )
            if st.button("🔄 Rebuild Index", key="is_rebuild"):
                st.cache_resource.clear()
                st.rerun()

    is_agent = get_interspeech_agent(years, filter_dementia, max_papers)
    chat_interface(
        agent=is_agent,
        messages=st.session_state.interspeech_messages,
        ask_func=ask_interspeech,
        input_key="is_input",
        placeholder="Ask a question about Interspeech research..."
    )

# ------------------------------------------------------------
# TAB 3: BIBLE ASSISTANT (PLACEHOLDER)
# ------------------------------------------------------------
with tab3:
    st.header("Bible Study Assistant")
    st.caption("Coming soon! This will allow you to ask questions about the Bible.")

    st.info("📖 The Bible assistant is not yet implemented. Check back later!")

    # Placeholder chat (same interface but with a dummy agent)
    def dummy_ask(agent, question):
        return "⚠️ This assistant is not yet implemented. Please select another tab."

    chat_interface(
        agent=None,
        messages=st.session_state.bible_messages,
        ask_func=dummy_ask,
        input_key="bible_input",
        placeholder="Ask a question about the Bible... (placeholder)"
    )