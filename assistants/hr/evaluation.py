"""Step 9: evaluate answer quality against a fixed set of test questions.

Unlike tracing (which just records what happened), evaluation runs the
agent against a known set of question/reference-answer pairs and scores
each answer using a second LLM as a judge. Results are uploaded to
LangSmith as a Dataset + Experiment, and a compact score table is
printed to the terminal.
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
from assistants.hr.pipeline import ask, build_hr_assistant
from assistants.hr import config as hr_config


logger = get_logger(__name__)

DATASET_NAME = "hr-policy-qna"

TEST_CASES = [
    {"question": "How many days of paid annual leave do I get per year?",
     "answer": "20 days"},
    {"question": "How many days of unused annual leave can be carried forward?",
     "answer": "Up to 5 days"},
    {"question": "How many paid sick days do I get per year?",
     "answer": "10 days"},
    {"question": "How many days per week can I work from home?",
     "answer": "Up to 2 days, with manager approval"},
    {"question": "How long is the probation period?",
     "answer": "3 months"},
    {"question": "What is the notice period during probation?",
     "answer": "15 days"},
    {"question": "What is the standard notice period for resignation?",
     "answer": "30 days"},
    {"question": "Within how many days must reimbursement claims be submitted?",
     "answer": "30 days of the expense"},
    {"question": "How many public holidays does the company observe each year?",
     "answer": "12"},
    {"question": "Within how many days is full and final settlement processed after the last working day?",
     "answer": "45 days"},
]


# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
def _get_judge_llm() -> ChatOpenAI:
    """Return a judge model routed through Portkey, same slug as the main app."""
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, provider=PRIMARY_PROVIDER)
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        model=config.JUDGE_MODEL_NAME,
    )


# ---------------------------------------------------------------------------
# Dataset bootstrap
# ---------------------------------------------------------------------------
def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist, and upload test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]},
             "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ],
    )
    return dataset


# ---------------------------------------------------------------------------
# Terminal score table
# ---------------------------------------------------------------------------
def _print_results_table(results) -> dict:
    """
    Print a compact per-question score table and return averages.

    Handles both LangSmith SDK shapes:
      - older: evaluation_results = {"correctness": [Eval...], "groundedness": [...]}
      - newer: evaluation_results = {"results": [Eval(key, score), ...]}
    where Eval is either an object or a dict.
    """
    rows = list(results)
    if not rows:
        print("  (no results returned)")
        return {}

    # ---------------- helpers ----------------
    def _iter_eval_results(r):
        """Yield EvaluationResult-like items, unwrapping whatever the SDK nests."""
        container = None
        if hasattr(r, "evaluation_results"):
            container = r.evaluation_results
        elif isinstance(r, dict):
            container = r.get("evaluation_results")

        if container is None:
            return

        # Newer SDK: {"results": [Eval, Eval, ...]}
        if isinstance(container, dict) and "results" in container:
            inner = container["results"]
            if isinstance(inner, list):
                for e in inner:
                    yield e
            else:
                yield inner
        # Older SDK: {"correctness": [Eval, ...], "groundedness": [...]}
        elif isinstance(container, dict):
            for v in container.values():
                if isinstance(v, list):
                    for e in v:
                        yield e
                else:
                    yield v
        elif isinstance(container, list):
            for e in container:
                yield e

    def _key_of(e):
        if hasattr(e, "key"):
            return e.key
        if isinstance(e, dict):
            return e.get("key")
        return None

    def _score_of(e):
        if hasattr(e, "score"):
            return e.score
        if isinstance(e, dict):
            return e.get("score")
        return None

    def _get_question(r):
        ex = None
        if hasattr(r, "example"):
            ex = r.example
        elif isinstance(r, dict):
            ex = r.get("example")

        if ex is None:
            if isinstance(r, dict):
                return (r.get("inputs") or {}).get("question", "?")
            return "?"

        # ex may be an object OR a dict
        if hasattr(ex, "inputs"):
            inputs = ex.inputs or {}
        elif isinstance(ex, dict):
            inputs = ex.get("inputs") or {}
        else:
            inputs = {}

        return (inputs or {}).get("question", "?")

    # ---------------- discover evaluator names ----------------
    evaluator_names = set()
    for r in rows:
        for e in _iter_eval_results(r):
            k = _key_of(e)
            if k:
                evaluator_names.add(k)
    evaluator_names = sorted(evaluator_names)

    if not evaluator_names:
        print("  (no evaluator scores found in SDK response; check LangSmith UI)")
        return {}

    # ---------------- header ----------------
    q_width = 52
    header = f"  {'question':<{q_width}}"
    for name in evaluator_names:
        header += f"  {name[:14]:>14}"
    print(header)
    print("  " + "-" * (q_width + 16 * len(evaluator_names)))

    # ---------------- rows ----------------
    totals = {name: 0.0 for name in evaluator_names}
    counts = {name: 0 for name in evaluator_names}

    for r in rows:
        q = _get_question(r)
        if len(q) > q_width - 2:
            q = q[: q_width - 2] + ".."
        line = f"  {q:<{q_width}}"

        row_scores = {}
        for e in _iter_eval_results(r):
            k = _key_of(e)
            s = _score_of(e)
            if k and s is not None:
                row_scores.setdefault(k, []).append(s)

        for name in evaluator_names:
            scores = row_scores.get(name, [])
            if scores:
                s = scores[0]
                line += f"  {s:>14.2f}"
                totals[name] += s
                counts[name] += 1
            else:
                line += f"  {'-':>14}"
        print(line)

    # ---------------- averages ----------------
    print("  " + "-" * (q_width + 16 * len(evaluator_names)))
    avg_line = f"  {'AVERAGE':<{q_width}}"
    averages = {}
    for name in evaluator_names:
        if counts[name]:
            avg = totals[name] / counts[name]
            averages[name] = avg
            avg_line += f"  {avg:>14.2f}"
        else:
            avg_line += f"  {'n/a':>14}"
    print(avg_line)

    return averages


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_evaluation():
    """
    Upload the dataset (if needed) and run the correctness + groundedness evaluation.

    Returns:
        (results, averages) where results is the ExperimentResults object
        and averages is {evaluator_name: avg_score}.
    """
    client = Client()
    dataset = _ensure_dataset(client)

    agent = build_hr_assistant()
    retriever = get_retriever(load_vector_store(hr_config), k=hr_config.HR_TOP_K)

    def target(inputs: dict) -> dict:
        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(chunk.page_content for chunk in chunks)
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
        experiment_prefix="hr-policy-eval",
        description="HR policy assistant correctness + groundedness evaluation",
    )

    # -------- local score table --------
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