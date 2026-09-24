"""Show the latest HR policy evaluation scores from LangSmith.

Run:
    python -u scripts/show_eval_scores.py
"""

import os
import sys

# Make the repo root importable, regardless of where the script is invoked from
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from langsmith import Client
from assistants.hr.evaluation import DATASET_NAME


def main():
    client = Client()

    # Find experiments attached to the dataset
    try:
        experiments = list(client.list_projects(reference_dataset_name=DATASET_NAME))
    except TypeError:
        # older SDK — filter manually
        experiments = [p for p in client.list_projects()
                       if (p.name or "").startswith("hr-policy-eval")]

    if not experiments:
        print(f"No experiments found for dataset '{DATASET_NAME}'.")
        print("Run:  python -u main_hr.py --eval")
        sys.exit(0)

    latest = sorted(experiments, key=lambda p: p.start_time or 0)[-1]
    print(f"Latest experiment : {latest.name}")
    print(f"Start time        : {latest.start_time}")
    print("-" * 68)

    runs = list(client.list_runs(project_name=latest.name, is_root=True))
    for run in runs:
        q = (run.inputs or {}).get("question", "?")
        fb = list(client.list_feedback(run_ids=[run.id]))
        scores = {f.key: f.score for f in fb}
        score_str = "  ".join(f"{k}={v:.2f}" for k, v in scores.items() if v is not None)
        print(f"  {q[:55]:<55}  {score_str}")

    print("-" * 68)
    print("Open https://smith.langchain.com for full reasoning traces.")


if __name__ == "__main__":
    main()