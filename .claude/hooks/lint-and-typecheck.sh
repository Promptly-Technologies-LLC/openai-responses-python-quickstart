#!/bin/bash
# Hook: Run linting and type checking at the end of a model workflow run
# Blocks Claude from stopping if there are errors, so Claude can fix them

set -o pipefail

# Read input from stdin
input=$(cat)

# Check if we're already in a stop hook continuation to prevent infinite loops
stop_hook_active=$(echo "$input" | jq -r '.stop_hook_active // false')

if [ "$stop_hook_active" = "true" ]; then
    # Already tried to fix once, don't block again to avoid infinite loops
    exit 0
fi

# Run linting
lint_output=$(uv run ruff check 2>&1)
lint_exit=$?

# Run type checking
type_output=$(uv run ty check 2>&1)
type_exit=$?

# If both passed, exit successfully
if [ $lint_exit -eq 0 ] && [ $type_exit -eq 0 ]; then
    exit 0
fi

# Build error message
errors=""

if [ $lint_exit -ne 0 ]; then
    errors="=== Linting Errors (ruff check) ===
$lint_output
"
fi

if [ $type_exit -ne 0 ]; then
    errors="${errors}
=== Type Checking Errors (ty check) ===
$type_output"
fi

# Output errors to stderr and exit with code 2 to block stop and show to Claude
echo "$errors" >&2
exit 2
