"""Streamlit chat app for the Interspeech Research Assistant.

Run with:  streamlit run app_interspeech.py
"""

import streamlit as st

from interspeech.pipeline import ask, build_interspeech_assistant

st.set_page_config(page_title="Interspeech Research Assistant", page_icon="🎤")
st.title("🎤 Interspeech Research Assistant")
st.caption("Ask questions about research presented at Interspeech conferences.")

# Initialize session state for rebuild trigger
if "force_rebuild" not in st.session_state:
    st.session_state.force_rebuild = False

# Settings sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    years = st.multiselect(
        "Years",
        options=[2020, 2021, 2022, 2023, 2024, 2025],
        default=[2025]
    )
    
    filter_dementia = st.checkbox("🧠 Only dementia-related papers", value=True)
    
    max_papers = st.number_input(
        "Max papers per year",
        min_value=0,
        max_value=500,
        value=20,
        help="0 = load all papers"
    )
    
    # Rebuild button
    if st.button("🔄 Rebuild Index", help="Re-scrape and rebuild the vector store"):
        st.session_state.force_rebuild = True
        st.cache_resource.clear()
        st.rerun()


@st.cache_resource(show_spinner="Building the research assistant...")
def get_agent(_years, _filter_dementia, _max_papers, _force_rebuild):
    """Build and cache the agent."""
    return build_interspeech_assistant(
        years=_years,
        filter_dementia=_filter_dementia,
        max_papers=_max_papers if _max_papers > 0 else None,
        rebuild=_force_rebuild  # <-- Now uses the actual button state!
    )


# Get the rebuild flag and reset it
force_rebuild = st.session_state.pop("force_rebuild", False)

# Get agent with current settings
agent = get_agent(years, filter_dementia, max_papers, force_rebuild)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
question = st.chat_input("Ask a question about Interspeech research...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            answer = ask(agent, question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
