"""Evaluation for the Interspeech research assistant.

Mirrors the HR evaluation: runs the agent against a fixed set of
question/reference-answer pairs, scores each answer with a judge LLM,
uploads to LangSmith, and prints a terminal score table.
"""

from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from core import config
from core.gateway import PRIMARY_PROVIDER
from core.logger import get_logger
from core.vector_store import get_retriever, load_vector_store
from assistants.interspeech import config as is_config
from assistants.interspeech.pipeline import ask, build_interspeech_assistant


logger = get_logger(__name__)

DATASET_NAME = "interspeech-paper-qna"


# ============================================================
# TEST CASES
# ============================================================
# NOTE: These reference answers are placeholders. Before running eval for
# real, populate them from your actual scraped data:
#
#   1. Run:  python -u scripts/explore_interspeech.py
#   2. It prints what the retriever actually finds for each question.
#   3. Copy the concrete facts you see into the "answer" fields below.
#
# Until you do that, the judge will score against incomplete references
# and you'll get artificially low correctness scores.
# ============================================================
TEST_CASES = [
    {"question": "What methods are used for dementia detection from speech?",
     "answer": "TODO: fill in from scraped data"},
    {"question": "Which papers discuss speech biomarkers for cognitive decline?",
     "answer": "TODO: fill in from scraped data"},
    {"question": "What datasets are commonly used in Alzheimer's speech research?",
     "answer": "TODO: fill in from scraped data"},
    {"question": "What features are extracted from speech for MCI detection?",
     "answer": "TODO: fill in from scraped data"},
    {"question": "Are there papers using deep learning for dementia classification?",
     "answer": "TODO: fill in from scraped data"},
]


def _get_judge_llm() -> ChatOpenAI:
    """Judge model via Portkey (same slug as the main app)."""
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, provider=PRIMARY_PROVIDER)
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        model=config.JUDGE_MODEL_NAME,
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
    """Compact per-question score table. Handles both SDK row shapes."""
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
        print("  (no evaluator scores found; check LangSmith UI)")
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
    """Upload dataset, run evaluation, print score table. Returns (results, averages)."""
    client = Client()
    dataset = _ensure_dataset(client)

    agent = build_interspeech_assistant()
    retriever = get_retriever(load_vector_store(is_config), k=is_config.INTERSPEECH_TOP_K)

    def target(inputs: dict) -> dict:
        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(c.page_content for c in chunks)
        return {"answer": answer, "context": context}

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
            context=outputs["context"],
        )

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix="interspeech-eval",
        description="Interspeech assistant correctness + groundedness evaluation",
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