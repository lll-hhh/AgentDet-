# IPIGuard Integration

AgentDET does not bundle the IPIGuard or AgentDojo repositories. The local
adapter expects an external checkout containing IPIGuard's AgentDojo source
tree.

## Setup

1. Clone the IPIGuard artifact outside this repository.
2. Check out the revision used by the benchmark:
   `4e686ed2f62c135cb12564a4466daa95f3ace878`.
3. Set `IPIGUARD_UPSTREAM_SRC` to the directory that contains the `agentdojo`
   Python package. In the upstream layout this is normally the AgentDojo
   `src` directory.

PowerShell example:

```powershell
$env:IPIGUARD_UPSTREAM_SRC = "<path-to-ipiguard-agentdojo-src>"
python run_eval.py --matrix-defenses ipiguard <other arguments>
```

The adapter validates the expected files before importing the upstream code.
No upstream benchmark suites, provider backends, model weights, or generated
artifacts are stored in AgentDET.
