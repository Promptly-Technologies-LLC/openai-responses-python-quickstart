---
description: Use when implementing any feature or bugfix, before writing implementation code - write the test first, watch it fail, write minimal code to pass; ensures tests actually verify behavior by requiring failure first
---

# Test-Driven Development (TDD)

## Overview

Write the test first. Watch it fail. Write minimal code to pass. No exceptions without your human partner's permission.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

**Before beginning work:** Understand core code and test infrastructure patterns if you haven't already.

## What if you've already written code before invoking this command?

Wrote code before the test? Delete or stash and pop it only after you've observed a failing test.

## Red-Green-Refactor

### RED - Write Failing Test

Write one minimal test showing what should happen.

```python
def test_rejects_empty_email():
    result = submit_form({'email': ''})
    assert result.error == 'Email required'
```

**Requirements:**
- Clear name
- Test one real behavior (mocks used for behavior isolation, not behavior emulation)
- Test should target the *desired* behavior, not the behavior the agent regards as achievable

### Verify RED - Watch It Fail

**MANDATORY. Never skip.**

```bash
uv run pytest tests/path/test.py::test_name -v
```

Success condition:
- Test fails (not errors) for the expected reason

### GREEN - Minimal Code

Write simplest code to pass the test.

```bash
uv run pytest tests/path/test.py::test_name -v
```

Confirm:
- Test passes
- Other tests still pass

**Test fails?** Fix code, not test.

**Other tests fail?** Fix now.

**Implementation not achievable?** Stop work immediately and ask for guidance. A failing test we can't turn green is a partial success condition, because it alerts us to architecture issues that need to be addressed before we can continue.

### REFACTOR - Clean Up

After green only:
- Remove duplication
- Improve names
- Extract helpers

Keep tests green. Don't add behavior.

## Verification Checklist

Before marking work complete:

- [ ] Every new function/method has a test
- [ ] Watched each test fail before implementing
- [ ] Each test failed for expected reason (feature missing, not typo)
- [ ] Wrote minimal code to pass each test
- [ ] All tests pass
- [ ] Edge cases and errors covered
