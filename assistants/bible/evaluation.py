"""Evaluation for the Bible assistant via LangSmith + OpenEvals."""

from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from core import config
from core.gateway import PRIMARY_PROVIDER
from core.logger import get_logger
from assistants.bible.pipeline import ask, build_bible_assistant

logger = get_logger(__name__)

DATASET_NAME = "bible-nwt-qna"

# Reference answers are drawn from the NWT. Update if you notice drift.
TEST_CASES = [
    {"question": "What does John 3:16 say?",
     "answer": "God loved the world so much that he gave his only-begotten Son, so that everyone exercising faith in him might not be destroyed but have everlasting life."},
    {"question": "What does Psalm 23:1 say?",
     "answer": "Jehovah is my Shepherd. I will lack nothing."},
    {"question": "What is the first verse of the Bible?",
     "answer": "In the beginning God created the heavens and the earth. (Genesis 1:1)"},
    {"question": "How many days did God take to create according to Genesis 1?",
     "answer": "Six creative days, then he rested on the seventh."},
    {"question": "What does 1 Corinthians 13:4 say about love?",
     "answer": "Love is patient and kind. Love is not jealous, it does not brag, does not get puffed up."},
    {"question": "What is the greatest commandment according to Jesus?",
     "answer": "Love Jehovah your God with your whole heart, soul, and mind; and the second, love your neighbor as yourself."},
]


def _get_judge_llm() -> ChatOpenAI:
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, provider=PRIMARY_PROVIDER)
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        model=config.JUDGE_MODEL_NAME,
        max_tokens=config.JUDGE_MAX_TOKENS,
    )


def _ensure_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": c["question"]},
             "outputs": {"answer": c["answer"]}}
            for c in TEST_CASES
        ],
    )
    return dataset


def _print_results_table(results) -> dict:
    rows = list(results)
    if not rows:
        print("  (no results returned)")
        return {}

    def _iter_eval_results(r):
        container = getattr(r, "evaluation_results", None)
        if container is None and isinstance(r, dict):
            container = r.get("evaluation_results")
        if container is None:
            return
        if isinstance(container, dict) and "results" in container:
            inner = container["results"]
            yield from (inner if isinstance(inner, list) else [inner])
        elif isinstance(container, dict):
            for v in container.values():
                yield from (v if isinstance(v, list) else [v])
        elif isinstance(container, list):
            yield from container

    def _key(e):
        return getattr(e, "key", None) or (isinstance(e, dict) and e.get("key"))

    def _score(e):
        return getattr(e, "score", None) if not isinstance(e, dict) else e.get("score")

    def _question(r):
        ex = getattr(r, "example", None)
        if ex is None and isinstance(r, dict):
            ex = r.get("example")
        if ex is None:
            return "?"
        inputs = getattr(ex, "inputs", None) or (isinstance(ex, dict) and ex.get("inputs")) or {}
        return (inputs or {}).get("question", "?")

    names = set()
    for r in rows:
        for e in _iter_eval_results(r):
            k = _key(e)
            if k:
                names.add(k)
    names = sorted(names)
    if not names:
        print("  (no evaluator scores found)")
        return {}

    qw = 52
    hdr = f"  {'question':<{qw}}"
    for n in names:
        hdr += f"  {n[:14]:>14}"
    print(hdr)
    print("  " + "-" * (qw + 16 * len(names)))

    totals = {n: 0.0 for n in names}
    counts = {n: 0 for n in names}

    for r in rows:
        q = _question(r)
        q = (q[:qw - 2] + "..") if len(q) > qw else q
        line = f"  {q:<{qw}}"
        row = {}
        for e in _iter_eval_results(r):
            k, s = _key(e), _score(e)
            if k and s is not None:
                row.setdefault(k, []).append(s)
        for n in names:
            scores = row.get(n, [])
            if scores:
                line += f"  {scores[0]:>14.2f}"
                totals[n] += scores[0]
                counts[n] += 1
            else:
                line += f"  {'-':>14}"
        print(line)

    print("  " + "-" * (qw + 16 * len(names)))
    avg_line = f"  {'AVERAGE':<{qw}}"
    averages = {}
    for n in names:
        if counts[n]:
            avg = totals[n] / counts[n]
            averages[n] = avg
            avg_line += f"  {avg:>14.2f}"
        else:
            avg_line += f"  {'n/a':>14}"
    print(avg_line)
    return averages


def run_evaluation():
    client = Client()
    dataset = _ensure_dataset(client)

    agent = build_bible_assistant()

    def target(inputs: dict) -> dict:
        answer = ask(agent, inputs["question"])
        return {"answer": answer, "context": ""}

    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=_get_judge_llm(),
    )
    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=_get_judge_llm(),
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        return groundedness_judge(
            outputs={"answer": outputs["answer"]},
            context=outputs.get("context") or " ",
        )

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix="bible-eval",
        description="Bible assistant correctness + groundedness evaluation",
    )

    print("\n" + "=" * 68)
    print(f"  Evaluation results — dataset '{DATASET_NAME}'")
    print(f"  Judge model: {config.JUDGE_MODEL_NAME}")
    print("=" * 68)
    averages = _print_results_table(results)
    print("=" * 68)
    print(f"  Full details : https://smith.langchain.com")
    print(f"  Project      : {config.LANGSMITH_PROJECT}")
    print(f"  Dataset      : {DATASET_NAME}")
    return results, averages