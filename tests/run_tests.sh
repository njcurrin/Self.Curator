#!/bin/bash
# CI entrypoint for self.curator test suite (self.ai tests only).
# Usage:
#   ./tests/run_tests.sh                    # run all self.ai tests
#   TEST_MARKERS=fast ./tests/run_tests.sh  # run only fast tests
#   TEST_MARKERS="not gpu" ./tests/run_tests.sh
#
# Invoke from host:
#   docker compose exec self-curator /app/tests/run_tests.sh

set -euo pipefail

cd /app

# Generate Parquet fixture if missing
if [ ! -f /app/tests/fixtures/selfai/sample_data.parquet ]; then
    python3 /app/tests/fixtures/selfai/generate_fixtures.py
fi

JUNIT_PATH="/app/tests/results/junit.xml"
mkdir -p "$(dirname "$JUNIT_PATH")"

# Build marker args as an array so multi-word expressions
# (e.g., TEST_MARKERS="not gpu") are passed as a single -m argument
# instead of word-splitting into separate pytest paths.
MARKER_ARGS=()
if [ -n "${TEST_MARKERS:-}" ]; then
    MARKER_ARGS=(-m "${TEST_MARKERS}")
fi

# Run only self.ai test sub-packages (not upstream NeMo Curator tests)
exec python3 -m pytest \
    tests/api/ \
    tests/nodes/ \
    tests/pipeline/ \
    tests/test_selfai_fixtures.py \
    "${MARKER_ARGS[@]}" \
    --junitxml="$JUNIT_PATH" \
    -v \
    "$@"
