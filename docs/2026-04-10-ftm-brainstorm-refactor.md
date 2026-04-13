# FTM-Brainstorm + Plan Template Refactor

## Problem

FTM-brainstorm does excellent research and user interaction, but its plan output lacks the buildability of superpowers:writing-plans. Specifically: no TDD workflow, no implementation code in plans, no intermediate spec document, frontend-biased wiring contracts, no self-review against spec, and research findings end up as ignorable hints instead of actual code.

This refactor closes those gaps while keeping everything FTM-brainstorm does better than superpowers (research sprints, AskUserQuestion, escalating depth, blackboard, pause/resume, discuss mode, YAGNI pressure, feature-type detection).

## Files to Modify

1. `~/.claude/skills/ftm-brainstorm/ftm-brainstorm.md` — main skill file
2. `~/.claude/skills/ftm-brainstorm/references/plan-template.md` — plan generation template

---

## Change 1: Add Spec Stage Between Phase 2 and Phase 3

**File:** `ftm-brainstorm.md`

**What:** After Phase 2 converges and the user approves the summary, write a spec document before generating the plan. Currently the flow is:

```
Phase 2 (research loop) → Phase 3 (plan generation)
```

Change to:

```
Phase 2 (research loop) → Phase 2.5 (spec document) → User Review → Phase 3 (plan from spec)
```

**Why:** The spec is a stable reference that the plan is validated against. Without it, all decisions live in the context register (in-memory) and get lost or muddled by Phase 3. Superpowers' two-stage pipeline (spec then plan) is why its plans have better internal consistency.

**How:** Add a new `PHASE 2.5: SPEC GENERATION` section between Phase 2 and Phase 3:

```markdown
# PHASE 2.5: SPEC GENERATION

After the user confirms the Phase 2 summary ("Here's what I think we've landed on"), write a spec document before proceeding to plan generation.

## Spec Document Structure

Save to `docs/specs/YYYY-MM-DD-<topic>.md` (or project-appropriate location):

```markdown
# [Feature Name] — Design Spec

## Summary
[2-3 sentences: what we're building and why]

## Architecture Decisions
[For each major decision: what was decided, why, what alternatives were rejected]

## Component Interfaces
[For each new component: class/function signatures, input/output types, registry keys]

## Data Flow
[How data moves through the system: inputs → transformations → outputs]

## Configuration Schema
[What config keys are needed, their types, defaults, and where they're read]

## Scope
### In v1
- [feature/component]
### Deferred
- [feature/component] — [why deferred]

## Known Risks
- [Component X] — [why risky, what could go wrong]
- [Assumption Y] — [what if wrong]

## Course/Domain Fidelity Notes
[For domain-specific work: what source material says, how implementation maps to it, known deviations]
```

## Spec Self-Review

After writing, check:
1. **Placeholder scan:** Any TBD, TODO, incomplete sections? Fix them.
2. **Internal consistency:** Do architecture decisions match component interfaces?
3. **Scope check:** Is this focused enough for a single plan?
4. **Ambiguity check:** Could any requirement be interpreted two ways? Pick one.

Fix issues inline. Then ask the user to review before proceeding to Phase 3.

## User Review Gate

> "Spec written to `<path>`. Please review — any changes before I generate the implementation plan?"

Wait for approval. Only proceed to Phase 3 after the user approves.
```

**Also update Phase 3** to reference the spec:

```markdown
# PHASE 3: PLAN GENERATION

Read `references/plan-template.md` for the full template and rules.

**CRITICAL:** Generate the plan FROM the spec document written in Phase 2.5. Every spec requirement must map to at least one task. Cross-reference during self-review.
```

**Update the session state section** to include spec path:

```
- **Phase 2.5**: spec document path, which sections written/approved, user review status
```

---

## Change 2: Rewrite Plan Template — TDD + Implementation Code

**File:** `references/plan-template.md`

**What:** Replace the current task structure (description + acceptance criteria + hints) with a TDD step structure that includes actual code.

**Current task structure (remove this):**

```markdown
### Task N: [Title]
**Description:** [What needs to be built]
**Files:** [Expected files to create/modify]
**Dependencies:** [Which tasks must complete first]
**Agent type:** [frontend-developer, backend-architect, etc.]
**Acceptance criteria:**
- [ ] [Specific, testable criterion]
**Hints:**
- [Relevant research finding with source URL]
**Wiring:**
  ...
```

**New task structure (replace with this):**

```markdown
### Task N: [Title]

**Files:**
- Create: `exact/path/to/file.py`
- Create: `tests/exact/path/to/test_file.py`
- Modify: `exact/path/to/existing.py`

**Dependencies:** [Task numbers, or "none"]
**Agent type:** [type]

- [ ] **Step 1: Write failing tests**

\```python
# tests/exact/path/to/test_file.py
# ACTUAL test code — not a description of what to test

def test_specific_behavior():
    result = function_under_test(specific_input)
    assert result == expected_output

def test_edge_case():
    ...
\```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/exact/path/to/test_file.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...'`

- [ ] **Step 3: Implement**

\```python
# exact/path/to/file.py
# ACTUAL implementation code — not pseudocode or stubs

def function_under_test(input):
    # real logic here
    return result
\```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/exact/path/to/test_file.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

\```bash
git add exact/path/to/file.py tests/exact/path/to/test_file.py
git commit -m "feat(task-N): concise description"
\```

**Wiring:**
  [domain-adaptive — see Change 3]

**Research context:**
- [Research finding adapted INTO the implementation above, with source URL]
- [Pitfall the implementation accounts for, with source URL]
```

**Add to Quality Rules section:**

```markdown
## No Placeholders (HARD RULE)

Every step must contain the actual content an engineer needs. These are **plan failures** — never write them:
- "TBD", "TODO", "implement later", "fill in details"
- "Add appropriate error handling" / "add validation" / "handle edge cases"
- "Write tests for the above" (without actual test code)
- "Similar to Task N" (repeat the code — the agent may read tasks out of order)
- Steps that describe what to do without showing how (code blocks required for code steps)
- References to types, functions, or methods not defined in any task
- "Implement the function" without the function body
- Acceptance criteria without corresponding test code

If a step changes code, SHOW THE CODE. If a step runs a command, SHOW THE COMMAND AND EXPECTED OUTPUT.
```

---

## Change 3: Make Wiring Contracts Domain-Adaptive

**File:** `references/plan-template.md`

**What:** Replace the current frontend-only wiring section with auto-detection based on project type.

**Replace the current Wiring Contract Rules section with:**

```markdown
## Wiring Contract Rules

Auto-detect project type from Phase 0 repo scan and use the matching wiring vocabulary:

### Python backend / data pipeline / strategy systems
```yaml
Wiring:
  registry_key: [key this component registers under, e.g., 'ichimoku_core']
  config_path: [YAML path this reads, e.g., 'strategies.ichimoku_fxaog.ichimoku']
  config_schema:
    key_name: {type: int, default: 26, description: "Kijun period"}
  data_flow_in: [what data this receives, from where, in what format]
  data_flow_out: [what data this produces, who consumes it]
  backward_compat: [old keys/interfaces preserved, or "N/A — new component"]
  imports: [what this file imports from the project]
  imported_by: [what files import from this one]
```

### Frontend (React/Vue/Svelte)
```yaml
Wiring:
  exports: [component/hook name]
  imported_by: [parent files]
  rendered_in:
    parent: [ParentComponent]
    placement: "[where in parent JSX]"
  route_path: [/path]
  nav_link:
    location: [sidebar|navbar|menu]
    label: "[Display text]"
  store_reads: [which state this reads]
  store_writes: [which state this writes]
```

### API / service
```yaml
Wiring:
  endpoint: [METHOD /path]
  auth: [mechanism]
  request_schema: [input type/shape]
  response_schema: [output type/shape]
  called_by: [which frontend/service calls this]
  calls: [which downstream services/DBs this touches]
```

### CLI tool
```yaml
Wiring:
  command: [command name and subcommands]
  config_file: [path and format]
  env_vars: [required/optional env vars]
  output_format: [stdout format]
  exit_codes: [meaning of each]
```

Use the vocabulary that matches the project. If a project spans multiple types (e.g., Python backend + React frontend), use the appropriate vocabulary per task.
```

---

## Change 4: Add Plan Self-Review Checklist

**File:** `references/plan-template.md`

**What:** Add a mandatory self-review section that cross-references the plan against the spec document from Phase 2.5.

**Add after the Quality Rules section:**

```markdown
## Plan Self-Review (mandatory — run before presenting to user)

After writing the complete plan, review it against the spec with fresh eyes:

1. **Spec coverage:** Open the spec document from Phase 2.5. For each requirement/decision, point to the task that implements it. If a spec requirement has no task, add one.

2. **Placeholder scan:** Search every task for: TBD, TODO, "implement later", "similar to Task N", steps without code blocks, function references not defined anywhere. Fix all of them.

3. **Type consistency:** Do class names, function names, method signatures, and config keys match across ALL tasks that reference them? If Task 3 defines `FractalEval` but Task 8 references `FractalEvaluator`, that's a bug. Fix it.

4. **Import chain:** For each new file, trace the import path from that file to the entry point (main script, test runner, config loader). Any missing intermediate imports or registration steps? Add them.

5. **Config completeness:** Is every config key referenced in implementation code also defined in the config task? Is every config key in the config task actually read by some component?

6. **Test coverage:** Does every new public function/class have at least one test in the plan? If not, add the test.

7. **Data flow completeness:** Can you trace data from input (e.g., OHLCV bars) through every transformation to output (e.g., trading signal)? Any gaps where "something feeds this but no task produces it"?

Fix issues inline. No need to re-review — just fix and move on.
```

---

## Change 5: Add Risk Notes to Plan Template

**File:** `references/plan-template.md`

**What:** Add a required Risk Notes section at the end of every plan.

**Add to the Plan Document Structure, after Execution Order:**

```markdown
## Risk Notes

Required section. Document known risks, untested assumptions, and complexity hotspots:

- **[Component/feature]** — [Why it's risky. What could go wrong. What's unvalidated.]
- **[Assumption]** — [What happens if this assumption is wrong. How to detect and recover.]
- **[Integration point]** — [What's untested. What could break when components connect.]
- **[Scope item]** — [Which features are most likely to be cut if time runs short. Impact of cutting them.]

This section helps the executing agent (and user) prioritize caution where it matters. At least 3 entries required.
```

---

## Change 6: Research Findings Must Feed Implementation Code

**File:** `ftm-brainstorm.md` (Phase 3 instructions) and `references/plan-template.md`

**What:** When research sprints find relevant code patterns, repos, or implementations, Phase 3 should adapt those findings INTO the implementation code blocks — not just list them as hints.

**Add to Phase 3 in ftm-brainstorm.md:**

```markdown
## Research-to-Code Bridge

When generating plan tasks, do NOT just list research findings as hints. Instead:

1. For each task, check the context register for research findings relevant to that task.
2. If a research sprint found a GitHub repo with a relevant implementation pattern, ADAPT that pattern into the Step 3 implementation code. Credit the source in a comment.
3. If a research sprint found a pitfall or failure mode, write the implementation code to HANDLE that pitfall. Note the source in a comment.
4. The "Research context" section of each task should explain HOW the research influenced the code — not just link to it.

Bad (hint that gets ignored):
```
**Hints:**
- See github.com/user/repo for Elliott wave implementation
```

Good (research adapted into code):
```
- [ ] **Step 3: Implement**
\```python
# Elliott wave counting adapted from github.com/user/repo/wave.py
# Key insight: hard constraints (W2 retrace, W3 shortest) must be
# checked BEFORE Fibonacci confidence scoring
def count_elliott(swings, direction):
    # Hard constraint checks first (auto-disqualify)
    if wave2_retraces_100pct(swings):
        return None
    ...
\```

**Research context:**
- Elliott hard constraints from github.com/user/repo — adapted into the disqualification checks above
- Fibonacci scoring approach from [blog post URL] — used for confidence calculation
```
```

**Update the Hints section in plan-template.md** — rename from "Hints" to "Research context" and change the rules:

```markdown
**Research context:**
- [How this research finding was incorporated into the code above, with source URL]
- [What pitfall the implementation handles, with source URL]
- "No specific research findings for this task" if nothing applies

Rules:
- Research findings must be ADAPTED into implementation code, not just linked
- Every URL cited should correspond to something visible in the code
- If research found a pattern that was rejected, note why: "Considered X from [URL] but rejected because Y"
```

---

## Verification

After making all 6 changes, verify by mentally running through a brainstorm session:

1. Phase 0 scans the repo (unchanged)
2. Phase 1 intake (unchanged)
3. Phase 2 research loop with AskUserQuestion (unchanged)
4. **NEW: Phase 2.5** writes spec document, user reviews and approves
5. Phase 3 generates plan FROM the spec, with:
   - TDD step structure (test → fail → implement → pass → commit)
   - Actual code in every step
   - Domain-adaptive wiring contracts
   - Research findings adapted into code
   - Risk notes section
   - Self-review checklist cross-referencing the spec
6. Plan presented incrementally, user approves, saved, handoff to ftm-executor

The output plan should now have both FTM's research depth AND superpowers' buildability.
