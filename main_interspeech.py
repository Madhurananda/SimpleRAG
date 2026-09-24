"""Command-line demo for the Interspeech Research Assistant.

Run with:  python main_interspeech.py
"""

from interspeech.pipeline import ask, build_interspeech_assistant


def main():
    print("🚀 Building Interspeech Research Assistant...")
    agent = build_interspeech_assistant(
        filter_dementia=True,
        max_papers=20,  # Limit for testing
        rebuild=False           # Set to True to force re-embedding (e.g., after changing CHUNK_SIZE)
    )
    print("Assistant ready!\n")

    demo_questions = [
        "What are the key ideas presented about dementia?",
        # "What methods are being used for Alzheimer's detection?",
        # "What is the state of research on speech biomarkers for cognitive decline?",
    ]

    for question in demo_questions:
        print("=" * 60)
        print("QUESTION:", question)
        print("-" * 60)
        answer = ask(agent, question, show_tokens=True)
        print("ANSWER:", answer)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
