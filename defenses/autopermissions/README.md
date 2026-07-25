# AutoPermissions-adapted baseline

This runtime defense adapts the IC-only permission prediction boundary from
"Towards Automating Data Access Permissions in AI Agents" (IEEE S&P 2026).
The official repository and data are distributed under CC BY 4.0:

- Paper: https://arxiv.org/abs/2511.17959
- Code: https://github.com/llm-platform-security/ai-agent-permissions
- DOI: https://doi.org/10.1109/SP63933.2026.00018

## AgentDET adaptation

The original work predicts whether a user permits an AI agent to share a data
type with a tool or service. AgentDET represents external effects as structured
tool calls, so this adaptation maps each sensitive tool call to one permission
request containing:

- trusted user task and benchmark domain;
- proposed tool and operation type;
- inferred data types;
- structured tool arguments;
- earlier permission decisions from the same run.

Predictions with confidence at or above `0.90` are applied automatically.
Lower-confidence predictions are escalated to the configured simulated user
policy (`oracle`, `allow`, or `reject`). This follows the paper's recommendation
to delegate uncertain permissions to users, but the paired-safe oracle is an
AgentDET evaluation mechanism rather than part of the upstream implementation.

The defense is runtime-only and does not use Gardener's two-stage planning and
leaf review.

## Run

```powershell
python run_eval.py `
  --matrix-models gpt-5 `
  --matrix-defenses autopermissions_hitl `
  --cases data `
  --tools config/tool_library.json `
  --output results\human_confirmation\autopermissions_gpt5 `
  --workers 4 `
  --save-traces `
  --api-key <KEY> `
  --base-url $env:OPENAI_BASE_URL
```
