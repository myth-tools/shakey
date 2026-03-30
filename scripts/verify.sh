#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# 🚀 Project Shakey — Industry-Grade Quality Verification [v1.1]
# ──────────────────────────────────────────────────────────────────────────────
#
# This script performs a localized, OS-aware audit of the entire workspace.
# It automatically detects platform capabilities to avoid feature-flag conflicts.
#
# Features:
#   - Automatic OS-Aware Feature Gating (Skips Metal on Linux)
#   - ANSI Colorized Status Reporting (respects NO_COLOR / non-TTY)
#   - Execution Timing for Performance Auditing
#   - Summary Dashboard with Health Statistics
#
# Usage:
#   ./scripts/verify.sh           # Run full audit
#   ./scripts/verify.sh --open    # Run full audit and open docs in browser
#
# ──────────────────────────────────────────────────────────────────────────────

# ── Bug Fix #5: Working-Directory Guard ──────────────────────────────────────
# Resolve the workspace root from the script's own location, so the script
# works correctly regardless of where the caller's CWD is.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$WORKSPACE_ROOT"

# ── Environment Configuration ─────────────────────────────────────────────────
set -e
export RUST_BACKTRACE=1

# ── Bug Fix #7: Terminal-Aware Color / NO_COLOR Support ───────────────────────
# Respect the NO_COLOR convention (https://no-color.org/) and suppress ANSI
# codes when stdout is not a TTY (e.g. piped to a log file).
# FORCE_COLOR=1 is only set when we actually have a real interactive terminal.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    export FORCE_COLOR=1
    BOLD="\033[1m"
    GREEN="\033[32m"
    BLUE="\033[34m"
    YELLOW="\033[33m"
    RED="\033[31m"
    MAGENTA="\033[35m"
    CYAN="\033[36m"
    NC="\033[0m"
else
    unset FORCE_COLOR
    BOLD="" GREEN="" BLUE="" YELLOW="" RED="" MAGENTA="" CYAN="" NC=""
fi

# Timing Helper
STAMP_START=$(date +%s)

# ── OS Detection ──────────────────────────────────────────────────────────────
OS_UPPER=$(uname -s | tr '[:lower:]' '[:upper:]')
case "$OS_UPPER" in
    LINUX*)   PLATFORM="Linux" ;;
    DARWIN*)  PLATFORM="macOS" ;;
    *)        PLATFORM="Unknown ($OS_UPPER)" ;;
esac

# ── Bug Fix #6: EXTRA_FLAGS as a proper Bash array ───────────────────────────
# Using an array instead of a plain string prevents word-splitting issues if
# flags with spaces are ever added (e.g. --features "feat-a feat-b").
if [[ "$PLATFORM" == "macOS" ]]; then
    EXTRA_FLAGS=(--all-features)
else
    EXTRA_FLAGS=()
fi


# ── Helper Functions ──────────────────────────────────────────────────────────

header() {
    echo -e "\n${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}${MAGENTA}  $1 ${NC}"
    echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
}

status() {
    local label=$1
    local color=$2
    local icon=$3
    echo -e "[ ${color}${icon}${NC} ] ${BOLD}${label}...${NC}"
}

report_success() {
    echo -e " [ ${GREEN}SUCCESS${NC} ]"
}

report_failure() {
    echo -e "\n${RED}${BOLD}❌ ERROR: Verification aborted. Check logs above.${NC}\n"
    exit 1
}

# ── Bug Fix #3: Safe run helper (set -e + &&/|| false-abort prevention) ───────
# With `set -e`, if `report_success` itself exits non-zero for any reason, the
# `|| report_failure` branch would incorrectly fire even though the cargo
# command succeeded. Using an explicit if/then/else fully prevents this.
run_step() {
    local description="$1"
    shift
    if "$@"; then
        report_success
    else
        report_failure
    fi
}


# ── Verification Pipeline ─────────────────────────────────────────────────────

echo -e "${BOLD}${CYAN}🦾 Project Shakey Sovereign Auditor [Platform: ${YELLOW}${PLATFORM}${BOLD}${CYAN}] [Root: ${YELLOW}${WORKSPACE_ROOT}${BOLD}${CYAN}]${NC}"

# Phase 0: Auto-Format (Apply Sovereign Style)
header "Phase 0: Auto-Format (Sovereign Style Enforcement)"

# Step 0a: Strip trailing whitespace from all .rs files.
# rustfmt has an internal bug where it refuses to format files with trailing
# whitespace on empty/non-empty lines. We must clean these first.
status "Stripping Trailing Whitespace (Pre-Flight Fix)" "${YELLOW}" "🧹"

# Bug Fix #1: Removed the erroneous `-print` action.
# `-print` and `-exec` are independent actions. Having both means `find`
# prints every filename to stdout BEFORE executing `sed`, cluttering output.
# Dropped `-print` so only the mutation action runs. The `find` command's own
# exit code (not sed's via `+`) is what set -e sees, but a failing sed within
# `-exec ... +` causes find to exit non-zero, so errors are now propagated.
find . -path ./target -prune -o -name "*.rs" -exec sed -i 's/[[:space:]]*$//' {} +
report_success

# Step 0b: Apply standard code formatting.
status "Applying Code Formatting (cargo fmt --all)" "${CYAN}" "🎨"
run_step "cargo fmt" cargo fmt --all

# Step 0c: Confirmation pass — verify everything is clean.
status "Verifying Format is Clean (Confirmation Pass)" "${CYAN}" "✅"
run_step "cargo fmt --check" cargo fmt --all -- --check
FMT_STATUS="${GREEN}AUTO-FIXED & VERIFIED${NC}"


# Phase 0.5: Workspace Audit (Orphan Crate Check)
header "Phase 0.5: Workspace Audit (Sovereign Integrity Check)"
status "Auditing Workspace Members in 'crates/'" "${YELLOW}" "📂"

# Bug Fix #4: Scope the grep to the `members = [...]` block only.
#
# The old `grep -q "crates/$crate_dir" Cargo.toml` matched ANY line in the
# file — including the [workspace.dependencies] path entries. A crate removed
# from `members` but kept as a dependency path would incorrectly pass the
# check. We now extract only the members block with awk before grepping.
MEMBERS_BLOCK=$(awk '/^members[[:space:]]*=/{found=1} found{print} /^\]/{if(found) exit}' Cargo.toml)

for d in crates/*/; do
    crate_dir=$(basename "$d")
    if ! echo "$MEMBERS_BLOCK" | grep -q "crates/$crate_dir"; then
        echo -e "${RED}❌ ERROR: Orphan crate detected! 'crates/$crate_dir' is NOT in Cargo.toml workspace members.${NC}"
        exit 1
    fi
done
report_success


header "Phase 1: Static Analysis & Type Checking"
status "Running Cargo Check (Workspace + All Targets)" "${BLUE}" "🔍"
run_step "cargo check" cargo check --workspace --all-targets

# Phase 2: Linting
header "Phase 2: Linting & Best Practices"
status "Running Cargo Clippy (Strict)" "${YELLOW}" "✨"
run_step "cargo clippy" cargo clippy --workspace --all-targets -- -D warnings

# Phase 3: Full Workspace Build
header "Phase 3: Full Workspace Build (Synthesis)"
status "Building Final Application Binaries & Benchmarks" "${MAGENTA}" "🏗️"
run_step "cargo build" cargo build --workspace --all-targets

# Phase 4: Runtime Validation (Unit & Integration)
header "Phase 4: Runtime Validation (Unit & Integration)"
# NOTE: --all-targets intentionally NOT used here.
# `cargo test --all-targets` includes bench targets, and Criterion's test
# harness executes the full benchmark loop when invoked via `cargo test`,
# causing the script to hang (observed: stuck on "WCSI Generation/Standard
# Autoregressive" for minutes+). Bench target *compilation* is already
# proven in Phase 3 (cargo build --workspace --all-targets). We only run
# actual test logic here: lib, bins, integration tests, and examples.
status "Running Unit & Integration Tests (lib + bins + tests + examples)" "${GREEN}" "🧪"
run_step "cargo test" cargo test --workspace --lib --bins --tests --examples

# Phase 4.5: Documentation Logic (Doc Tests)
header "Phase 4.5: Documentation Logic (Doc Tests)"
status "Running Doc Tests (Exhaustive)" "${CYAN}" "📚"
run_step "cargo test --doc" cargo test --workspace --doc


# Phase 5: API Documentation Audit
header "Phase 5: API Documentation Audit"
status "Building Detailed Project Report (Inc. Private Items)" "${MAGENTA}" "📖"

# Bug Fix #2: Removed `--all-targets` from `cargo doc`.
# `cargo doc` does NOT accept `--all-targets`; that flag is only valid for
# check/build/clippy/test. Passing it causes a hard error on cargo ≥1.76.
# Bug Fix #6 (applied): EXTRA_FLAGS expanded as "${EXTRA_FLAGS[@]}" (array).
run_step "cargo doc" cargo doc --workspace --no-deps --document-private-items "${EXTRA_FLAGS[@]}"



# ── Final Summary Dashboard ───────────────────────────────────────────────────
STAMP_END=$(date +%s)
DURATION=$((STAMP_END - STAMP_START))

# Optional Capability Check
command -v gnuplot >/dev/null 2>&1 && GNUPLOT_READY="${GREEN}YES${NC}" || GNUPLOT_READY="${YELLOW}NO (Install for better plots)${NC}"

# Generate interactive documentation URL (ANSI Hyperlink)
DOC_URI="file://${WORKSPACE_ROOT}/target/doc/shakey_core/index.html"
DOC_LINK="\e]8;;${DOC_URI}\a${CYAN}${BOLD}[ Open Project Report ↗ ]${NC}\e]8;;\a"

echo -e "\n${BOLD}${MAGENTA}📊 Exhaustive Verification Dashboard:${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
printf "  %-30s %b\n" "Platform Status:"          "${GREEN}SOVEREIGN (${PLATFORM})${NC}"
printf "  %-30s %b\n" "Workspace Root:"            "${CYAN}${WORKSPACE_ROOT}${NC}"
printf "  %-30s %b\n" "Workspace Health:"          "${GREEN}100% PROTECTED (All Targets Verified)${NC}"
printf "  %-30s %b\n" "Formatting Status:"         "${FMT_STATUS}"
printf "  %-30s %b\n" "Doc Test Integrity:"        "${GREEN}VERIFIED${NC}"
printf "  %-30s %b\n" "Bench Capability:"          "${GNUPLOT_READY}"
printf "  %-30s %b\n" "Documentation:"             "${GREEN}GENERATED (Detailed Report Ready)${NC}"
printf "  %-30s %b\n" "Doc Access URL:"            "${DOC_LINK}"
printf "  %-30s %b\n" "Total Audit Time:"          "${DURATION} seconds"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "\n${BOLD}${GREEN}✅ ALL SYSTEMS NOMINAL. Project Shakey is fully verified for production.${NC}"


# ── Bug Fix #8: Precise --open flag parsing ───────────────────────────────────
# The old `[[ "$*" == *"--open"* ]]` substring match would falsely match
# any argument containing "--open" (e.g. --reopen, --no-open). We now iterate
# positional parameters and require an exact string match.
OPEN_REPORT=false
for arg in "$@"; do
    [[ "$arg" == "--open" ]] && OPEN_REPORT=true
done

open_docs() {
    status "Launching Detailed Report" "${CYAN}" "🌐"
    if [[ "$PLATFORM" == "macOS" ]]; then
        open "$DOC_URI" >/dev/null 2>&1
    else
        xdg-open "$DOC_URI" >/dev/null 2>&1
    fi
}

if $OPEN_REPORT; then
    open_docs
elif [[ -t 0 ]]; then
    echo -ne "  ${YELLOW}PROMPT: Press ${BOLD}[O]${NC}${YELLOW} to open report in browser, or any other key to exit... ${NC}"
    read -n 1 -r -t 10 RESPONSE
    echo ""
    if [[ "$RESPONSE" =~ ^[oO]$ ]]; then
        open_docs
    fi
else
    echo -e "💡 ${CYAN}Tip: Run 'scripts/verify.sh --open' for automated report launching.${NC}\n"
fi
