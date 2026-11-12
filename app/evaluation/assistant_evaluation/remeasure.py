"""
Set-up of the re-measurement (V25 addendum): a second data directory next to
the first measurement's, which is only read.

seed() copies into design.V25_DIR what the revised routing instruction cannot
change and that the addendum says to reuse rather than pay again:

- sets/requests.json and sets/checks.json (A, B, D, E and F, unchanged);
- runs/P.jsonl and runs/depth_check.jsonl (the reference P has no LLM, and
  the retrieval code did not change between the two measurements);
- query_vectors/ (vectors are stored by text; new texts are added to the copy).

Judgments are not copied: judging.prior_caches() reads V14's and the first
measurement's caches, and new pairs go to the second directory's own caches.
The simulated user's calls are not copied either: they depend on the
assistant's question, which may change.

seed() also writes instructions.json: the routing instruction of each
measurement, read from git at the commit each was run with (the first at
design.FIRST_RUNS_COMMIT, the second at design.INSTRUCTION_COMMIT), so the
report can show both without importing old code.
"""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from app.evaluation.assistant_evaluation import design as D

PROMPTS_MODULE = "app/infrastructure/llm/prompts_assistant.py"
INSTRUCTIONS_FILE = "instructions.json"
COPIED = ("sets/requests.json", "sets/checks.json", "runs/P.jsonl", "runs/depth_check.jsonl",
          "query_vectors/qwen__qwen3-embedding-8b.jsonl")


def _module_constants(source: str) -> Dict[str, str]:
    """String constants assigned at module level in a Python source (no import, no execution)."""
    out: Dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def instruction_at(commit: str) -> Dict[str, Any]:
    """The routing instruction of the prompts module at `commit` (git show, read only)."""
    source = subprocess.run(["git", "show", f"{commit}:{PROMPTS_MODULE}"], check=True, capture_output=True,
                            text=True).stdout
    constants = _module_constants(source)
    prompt = constants["ROUTER_SYSTEM_PROMPT"]
    return {"commit": commit, "prompts_version": constants.get("ASSISTANT_PROMPTS_VERSION"),
            "router_system_prompt": prompt,
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
            "route_lines": [line for line in prompt.splitlines() if line.startswith('- "')]}


def seed(first_dir: Path = D.DATA_DIR, second_dir: Path = D.V25_DIR, log=print) -> Dict[str, Any]:
    """Copy the reusable files (only those not yet there) and write instructions.json."""
    copied = []
    for name in COPIED:
        target = second_dir / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(first_dir / name, target)
        copied.append(name)
    instructions = {"first": instruction_at(D.FIRST_RUNS_COMMIT), "second": instruction_at(D.INSTRUCTION_COMMIT)}
    (second_dir / INSTRUCTIONS_FILE).write_text(json.dumps(instructions, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    log(f"seeded {second_dir}: copied {copied or 'nothing new'}; instructions "
        f"{instructions['first']['prompts_version']} ({D.FIRST_RUNS_COMMIT}) and "
        f"{instructions['second']['prompts_version']} ({D.INSTRUCTION_COMMIT})")
    return {"copied": copied, "instructions": instructions}


def current_instruction_matches(second_dir: Path = D.V25_DIR) -> bool:
    """True if the code's routing instruction is the one recorded for the second measurement."""
    from app.infrastructure.llm.prompts_assistant import ROUTER_SYSTEM_PROMPT

    recorded = json.loads((second_dir / INSTRUCTIONS_FILE).read_text(encoding="utf-8"))["second"]
    return recorded["router_system_prompt"] == ROUTER_SYSTEM_PROMPT
