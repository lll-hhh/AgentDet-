# Dataset

AgentDET contains 94 safe cases and 549 unsafe variants, for 643 evaluations.
Safe cases are deduplicated by `case_id`; unsafe variants are counted by their
pair run identifier.

## Domains

| Domain | Safe | Unsafe | Total |
|---|---:|---:|---:|
| Finance | 15 | 110 | 125 |
| Medical | 11 | 66 | 77 |
| Office | 11 | 61 | 72 |
| Shopping | 17 | 77 | 94 |
| SmartHome | 11 | 57 | 68 |
| System | 17 | 111 | 128 |
| Travel | 12 | 67 | 79 |
| **Total** | **94** | **549** | **643** |

## Release Integrity Corrections

The public artifact applies three metadata/contract corrections discovered by
automated validation:

- duplicate entries were removed from six manifests without removing case files;
- `request_specialist_referral` was restored to five unsafe case tool lists
  where the action already appeared in ground truth;
- the existing read-only `read_referral_note` implementation was restored to
  the active tool catalog and defense policy maps.

These corrections do not change task text, injected payloads, expected states,
or the 94/549 case denominators. They make every released ground-truth action
available through the corresponding case-level tool contract.

## Evidence Delay

The case taxonomy maps `evidence_delay=1/2/3` to LagA/LagB/LagC.

| Delay | Safe | Unsafe | Total |
|---|---:|---:|---:|
| LagA | 4 | 28 | 32 |
| LagB | 46 | 294 | 340 |
| LagC | 44 | 227 | 271 |
| **Total** | **94** | **549** | **643** |
