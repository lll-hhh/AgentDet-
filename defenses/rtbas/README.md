# RTBAS benchmark adaptation

This directory implements the mechanism described in RTBAS
(`arXiv:2502.08966`) directly against AgentDET's planner/runtime plugin APIs.
The paper does not publish an official artifact, so this is an independent
reimplementation rather than imported upstream code.

## Execution flow

1. Label system/user regions as trusted-public.
2. Label tool outputs using the static policy in `config/rtbas_policy.json`.
3. Run an LM-judge dependency screener before each agent generation.
4. Join labels of relevant regions and redact more restrictive irrelevant
   regions from the generation history.
5. Check every proposed tool call against its information-flow policy.
6. When the policy is violated, simulate user confirmation with one of:
   `oracle`, `allow`, or `reject`.
7. A denied call is skipped and represented by a synthetic tool result so the
   agent can replan without executing the side effect.

The default `oracle` policy approves only calls that exactly match the paired
safe case's ground truth. This is an AgentDET evaluation adaptation. The RTBAS
paper's AgentDojo experiment did not model a user and skipped unauthorized
calls instead.

## Run

```powershell
python run_eval.py `
  --matrix-models gpt-5 `
  --matrix-defenses rtbas `
  --cases data `
  --tools config/tool_library.json `
  --output results\human_confirmation\rtbas_gpt5 `
  --workers 4 `
  --save-traces `
  --api-key <KEY> `
  --base-url $env:OPENAI_BASE_URL
```

For a lower-cost deterministic smoke run, set `planner_config.screener` to
`lexical` in a YAML evaluation config.
