# AgentDET

AgentDET is a benchmark and evaluation harness for measuring how tool-using
LLM agents behave under indirect prompt injection, delayed evidence, and
conflicting instructions. It contains the released AgentDET case set, a
deterministic seven-domain tool simulator, adapters for existing defenses, the
Gardener confirmation defense, and an extensible adaptive-attack interface.

The repository is intended to be a clean research artifact: raw experiment
traces, private API keys, local model caches, and manuscript-only generated
tables are excluded from version control.

## Dataset

The released benchmark contains 643 paired evaluations:

- 94 safe tasks;
- 549 unsafe task and injection pairs;
- seven domains: Finance, Medical, Office, Shopping, SmartHome, System, and Travel;
- three evidence-delay groups: LagA, LagB, and LagC.

The active cases live under `data/`. See [data/README.md](data/README.md) for
domain counts, evidence-delay counts, and case-format details.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to a local `.env` or export credentials in the shell.
Never commit API keys.

PIGuard and ProtectAI download their public Hugging Face checkpoints on first
use. Model weights and caches are intentionally not stored in this repository.
IPIGuard and the optional CaMeL upstream interpreter are loaded from external
checkouts. Their adapter READMEs document the required upstream revisions.

## Quick Start

Run the undefended baseline:

```bash
python run_eval.py --llm gpt-5 --cases data --tools config/tool_library.json
```

Run one defense:

```bash
python run_eval.py --llm gpt-5 --defense fides --cases data \
  --tools config/tool_library.json --output results/fides_gpt5
```

Run a defense matrix:

```bash
python run_eval.py --matrix-models gpt-5 --matrix-defenses \
  sandwiching,spotlighting,instructional,piguard,protectai,fides,ipiguard,camel,drift,tool_filter \
  --cases data --tools config/tool_library.json --output results/defense_matrix
```

## Defenses

Implemented defense adapters live under `defenses/`:

- input/context defenses: Sandwiching, Spotlighting, Instructional;
- action/runtime defenses: PIGuard, ProtectAI, FIDES;
- planning defenses: IPIGuard, CaMeL, DRIFT, ToolFilter;
- human-confirmation baselines: RTBAS and AutoPermissions;
- Gardener: the benchmark-native confirmation defense.

## Adaptive Attacks

`adaptive_attacks/` exposes an AgentDojo-style method registry. Attack methods
receive the current task, injection sites, public defense profile, and
sanitized execution feedback, and return bounded payload replacements.

See [adaptive_attacks/README.md](adaptive_attacks/README.md) for the extension
contract and a minimal plugin example.

The interface is a supported extension point. External attack methods can be
loaded from a Python module without editing AgentDET.

## Repository Layout

```text
adaptive_attacks/   Adaptive attack API and methods
config/             Tool catalog and defense configurations
core/               Agent runtime and plugin orchestration
data/               Released AgentDET benchmark cases
defenses/           Defense adapters and Gardener implementation
evaluation/         Case-level evaluation and metrics
hooks/              Input and reasoning hooks
llm_backends/       Model provider adapters
tools/              Seven-domain tool simulator
```

## Validation

Run a syntax check after installation:

```bash
python -m compileall adaptive_attacks core defenses evaluation hooks llm_backends tools run_eval.py
```

Generated metrics, traces, and manuscript tables are intentionally excluded
from version control. Dataset-derived slice metadata is stored in
`data/lag_case_split.csv`.

## Third-Party Components

AgentDET uses benchmark-native adapters for published defenses. Upstream
repositories and model weights are not bundled. IPIGuard and CaMeL are loaded
from external checkouts; PIGuard and ProtectAI load their public model
checkpoints; RTBAS is independently reimplemented from its paper; and
AutoPermissions adapts the CC BY 4.0 `ai-agent-permissions` mechanism to
structured AgentDET tool calls. Upstream components remain subject to their
own licenses.

## License

AgentDET is released under the MIT License. Third-party components remain under
their own licenses.
