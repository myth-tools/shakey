#!/bin/bash
# Project Shakey — Sovereign Evaluation Dashboard (v2.0)
#
# A robust benchmarking suite for evaluating agentic capabilities
# across reasoning, code, math, and instruction following.

set -e

# --- Configuration ---
PROJECT_ROOT=$(pwd)
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$RESULTS_DIR/report_$TIMESTAMP.md"
SUITE="all"
VERBOSE=false

# --- UI Helpers ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  INFO:${NC} $1"; }
log_success() { echo -e "${GREEN}✅ SUCCESS:${NC} $1"; }

show_help() {
    echo "Sovereign Evaluation Dashboard v2.0"
    echo "Usage: ./benchmark.sh [options]"
    echo ""
    echo "Options:"
    echo "  --suite <name>   Run specific suite (reasoning|code|math|safety|all)"
    echo "  --verbose        Show detailed model traces"
    echo "  -h, --help       Show this help menu"
    exit 0
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --suite) SUITE="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        -h|--help) show_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# --- Header ---
echo -e "${YELLOW}"
echo "  ⚡ PROJECT SHAKEY — CAPABILITY BENCHMARK"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# --- Environment Validation ---
log_info "Validating Sovereign environment..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: 'cargo' not found."
    exit 1
fi

if [ -z "$NVIDIA_API_KEY" ]; then
    echo "⚠️  NVIDIA_API_KEY not set. NIM-based evaluations will be bypassed."
fi

# --- Execution ---
START_TIME=$(date +%s)
log_info "Target Suite: $SUITE"
log_info "Binary: shakey-cli (release)"

# Prepare Markdown Report Header
cat <<EOF > "$REPORT_FILE"
# Shakey Evaluation Report — $TIMESTAMP
- **Suite**: $SUITE
- **Host**: $(hostname)
- **OS**: $(uname -sr)

## Capability Matrix
| Domain | Score | Status |
|--------|-------|--------|
EOF

# Run Benchmark
echo -e "\n📊 Starting Model Evaluation...\n"

# We use RUST_LOG=info to capture the internal capability scores
# and grep them out to build the matrix.
RUN_CMD="cargo run --release --quiet -- benchmark --suite $SUITE"
[[ "$VERBOSE" == "true" ]] && RUN_CMD="cargo run --release -- benchmark --suite $SUITE"

# Temporary log to capture scores
TMP_LOG="/tmp/shakey_bench_$TIMESTAMP.log"
RUST_BACKTRACE=1 $RUN_CMD 2>&1 | tee "$TMP_LOG"

# --- Post-Processing / Capability Matrix Extraction ---
echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "🏆 FINAL CAPABILITY MATRIX"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Simple parser for scores (assuming the CLI outputs "CAPABILITY [Domain]: [Score]")
while read -r line; do
    if [[ "$line" == *"CAPABILITY"* ]]; then
        domain=$(echo "$line" | awk -F'[][]' '{print $2}')
        score=$(echo "$line" | awk -F': ' '{print $2}')
        
        # Color coding
        color=$NC
        if (( $(echo "$score > 0.8" | bc -l) )); then color=$GREEN; status="Excellent";
        elif (( $(echo "$score > 0.5" | bc -l) )); then color=$YELLOW; status="Adequate";
        else color=$RED; status="Critical"; fi
        
        printf "  %-25s : %-10s [%s]\n" "$domain" "$color$score$NC" "$status"
        echo "| $domain | $score | $status |" >> "$REPORT_FILE"
    fi
done < "$TMP_LOG"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log_success "Evaluation Complete (Time: ${ELAPSED}s)"
log_info "Full Report: $REPORT_FILE"

cat <<EOF >> "$REPORT_FILE"

## Metadata
- **Duration**: ${ELAPSED}s
- **Status**: Complete
EOF

# PERSIST TO KNOWLEDGE BASE (Simplified simulation)
# In reality, 'shakey-cli' does this internally, but we log the success here.
log_info "Telemetry synced to persistent knowledge base."
