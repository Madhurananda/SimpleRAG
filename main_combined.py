"""Interactive CLI for the Interspeech Research Assistant.

Run with:  python main_interspeech_interact.py
"""

from interspeech.pipeline import ask, build_interspeech_assistant


def main():
    print("🚀 Building Interspeech Research Assistant...")
    agent = build_interspeech_assistant(
        filter_dementia=True,
        max_papers=20,         # Adjust as needed
        rebuild=False           # Set to True to force re-embedding (e.g., after changing CHUNK_SIZE)
    )
    print("\n✅ Assistant ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("❓ Ask a question: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        print("🤔 Thinking...")
        # Set timeout to 10 seconds
        answer = ask(agent, question, timeout_seconds=10)
        print(f"\n🤖 Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
