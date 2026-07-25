# Adaptive Attack Interface

AgentDET exposes a registry-based attack interface modeled after AgentDojo. A
plugin receives one `AdaptiveAttackContext` containing:

- `user_task`: the complete AgentDET `Case`;
- `injection_task`: attacker goal, goal type, injection points, and target state;
- `target_pipeline`: the live victim executor/pipeline adapter;
- `tool_definitions`: full schemas for tools available in the case;
- `injection_candidates`: resolved untrusted fields that may be replaced;
- `ground_truth`: expected tool calls for the user task;
- `defense_profile`, previous coarse `feedback`, and the current iteration.

The built-in `pair_trajectory_rewrite` method additionally receives a sanitized
version of the previous execution trajectory: prompts, assistant/tool messages,
tool calls, final response, blocking result, and observable defense verdicts.
It never receives model reasoning, weights, gradients, logits, benchmark GT,
oracle matches, confirmation answers, or internal uncertainty scores.

New attacks should subclass `AdaptiveAttackMethod`, implement `attack(context)`,
and register the class with `@register_attack_method`. Existing attacks that
implement the older `generate(...)` method remain supported.

```python
from adaptive_attacks import (
    AdaptiveAttackContext,
    AdaptiveAttackMethod,
    AttackCandidate,
    register_attack_method,
)


@register_attack_method
class ExampleAttack(AdaptiveAttackMethod):
    name = "example"

    def attack(self, context: AdaptiveAttackContext) -> AttackCandidate:
        goal = context.injection_task.goal
        replacements = {
            site.site_id: f"{site.safe_text}\nAdditional record: {goal}"
            for site in context.get_injection_candidates()
        }
        return AttackCandidate(replacements=replacements, strategy="example")
```

External plugins can import this package and register methods without editing
the AgentDET core. `adaptive_attacks.registry` exposes method discovery and
construction, while `adaptive_attacks.runner` provides the reusable execution
pipeline.

`injection_candidates` are resolved from the benchmark's declared injection
points and the safe/unsafe case pair. They identify the concrete data fields on
the task path; plugins do not need to search or mutate unrelated environment
state.
