---
description: Use when creating new commands, editing existing commands, or verifying commands work before deployment - applies TDD to process documentation by testing with subagents before writing, iterating until bulletproof against rationalization (project)
---

# Writing Commands

## Overview

**Testing commands is just TDD applied to process documentation.**

Write commands that specify in positive terms what the agent should do in a way that is hard to misinterpret. Prefer clarifying positive instructions or including instructive examples to avoid the purple elephant problem. (If all else fails, we can guard against failure modes by building `.claude/hooks`, custom `ruff` lint rules, or other automated QA checks to steer the agent away from them at runtime.)

Use TDD to ensure that agents invoke commands when appropriate and that commands successfully extend their capabilities. Run a scenario without the command (RED - watch agent fail), write the command to make the desired behavior explicit (GREEN - watch agent comply), then refine wording to eliminate ambiguity and loopholes (REFACTOR - stay compliant).

**Core principle:** If you didn't watch an agent fail without the command, you don't know if the command adds real functionality.

## Command Structure

**File location (project commands):** `.claude/commands/<command-name>.md`
**Invocation:** `/<command-name> [arguments]`

Commands are Markdown files with a YAML frontmatter header. The **`description`** field is required for the command to show up cleanly in `/help` as a project command.

```markdown
---
description: Use when [trigger] - [what it does] (project)
argument-hint: [optional arguments summary]
# Required if injecting Bash output into the command prompt:
# allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(pytest:*)
---

# Command Title

## Overview
[Core principle in 1-2 sentences]

## Project-Specific Context
[How this command relates to Excel extraction, models, testing patterns]
[References to other project commands to consult]
[Optionally inject Bash output by prefixing with an exclamation mark, like "!`git status`"]

## When to Use
[Triggers and conditions]

## The Process
[Step-by-step instructions]

## Common Mistakes
[What NOT to do]

## Integration with Other Commands
[Which commands this requires or complements]
```

### Namespacing (optional)

If you want commands grouped in `/help`, put them in subdirectories under `.claude/commands/`. The **command name remains the filename**, but the subdirectory appears in the help description.

Example:
- `.claude/commands/frontend/component.md` creates `/component` and shows as `(project:frontend)`

## TDD Mapping for Command Testing

| TDD Phase | Command Testing | What You Do |
|-----------|---------------|-------------|
| **RED** | Baseline test | Run scenario WITHOUT command, watch agent fail |
| **Verify RED** | Capture rationalizations | Document exact failures verbatim |
| **GREEN** | Write command | Make the desired behavior explicit and testable |
| **Verify GREEN** | Pressure test | Run scenario WITH command, verify compliance |
| **REFACTOR** | Plug holes | Find new rationalizations, add counters |
| **Stay GREEN** | Re-verify | Test again, ensure still compliant |

## RED Phase: Baseline Testing

**Goal:** Run test WITHOUT the command - watch agent fail, document exact failures.

**Process:**
- [ ] Create a hard example task the command would help with
- [ ] Run task without the command and watch the agent fail
- [ ] Document failure modes and reasons for them in detail

Remember: we want the agent to correctly invoke the command based on the task description and project instruction files, so task prompt should omit any mention of commands.

## GREEN Phase: Write Minimal Command

Write a command addressing the specific baseline failures you documented. Add clarity to the command by specifying:

- Inputs (what information the agent should use)
- Outputs (what artifacts it should produce)
- Constraints (what "done" means)
- Decision rules (how to choose among options)
- Principled reasons (how the skill serves the workflow or development philosophy)

Run the same scenarios WITH the command. Agent should now comply.

## REFACTOR Phase: Close Loopholes

Agent violated rule despite having the command? Preempt rationalizations with positive instructions and explanations for why the rule is necessary:
- Agent rationalization: "This case is different because..."
- Language adjustment: "Always follow this rule, without exception, because it helps us catch problems early..."

## Testing Checklist

Before deploying the command:

**RED Phase:**
- [ ] Created a hard example task the command would help with
- [ ] Ran scenarios WITHOUT command (baseline)
- [ ] Documented agent failures

**GREEN Phase:**
- [ ] Wrote command addressing specific baseline failures
- [ ] Ran scenarios WITH command
- [ ] Agent now complies

**REFACTOR Phase:**
- [ ] Identified NEW failure modes from testing
- [ ] Updated command language to eliminate ambiguity
- [ ] Re-tested - agent still complies
