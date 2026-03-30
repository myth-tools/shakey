# Shakey Kaggle Master Pipeline — Sovereign Hardened v4

This is the definitive, fully hardened, production-ready Kaggle training script.
Every known failure mode has been addressed.

> [!IMPORTANT]
> **Pre-Flight Checklist (Do these in Kaggle BEFORE running):**
>
> 1. **Accelerator**: Set to **None (CPU only)**. BitNet 1.58-bit is optimized for CPU.
> 2. **Internet Access**: Toggle to **ON** (required for Rust install + NIM API + HF sync).
> 3. **Session type**: Set to **12-hour** (not interactive) for autonomous training.
> 4. **Kaggle Secrets** — Add ALL three:
>    - `NVIDIA_API_KEY` → your NVIDIA NIM API key (e.g. `nvapi-...`)
>    - `HF_TOKEN` → your Hugging Face **Write** token (e.g. `hf_...`)
>    - `HF_REPO_ID` → your HF repo (e.g. `Sishv/Shakey-Checkpoints`)
> 5. **GitHub Repo**: Confirm `GITHUB_REPO = "myth-tools/shakey"` below.
> 6. **Run all cells** or paste the entire block into a single Kaggle code cell.

---

### The Master Cell (Single Cell — Paste into Kaggle)

```python
# ==============================================================================
# SHAKEY SOVEREIGN KAGGLE PIPELINE — v4.0
# Fully hardened: anti-idle, HF checkpoint restore, progress dashboard,
# path-correct watcher, timeout guards, and graceful shutdown.
#
# v4 fixes (all 10 audited bugs):
#   - snapshot_download is a module-level fn, not an HfApi method
#   - allow_patterns uses ** for recursive glob
#   - Removed "--" separator that broke Clap subcommand parsing
#   - HF_REPO_ID None guard added to finally block
#   - process NameError risk removed (locals().get pattern)
#   - SIGTERM exit code handled generically for all signals
#   - Upload queue redesigned to 3-tuple with explicit path_in_repo
#   - uploaded_ids uses item.name (not abs path) for stable dedup
#   - finish() uses queue.join() to eliminate drain race condition
#   - SHAKEY_DATA_DIR is now honoured even in Kaggle mode (via env.rs)
# ==============================================================================

# ── 0. ANTI-IDLE BYPASS (Must be first) ───────────────────────────────────────
# NOTE: The real anti-idle mechanism is the live stdout stream in Step 8.
# The Rust binary (with --kaggle) prints a progress line every 60s, which
# keeps the Kaggle cell active. The JS below is a best-effort supplemental
# guard; DOM clicks do not directly reset Kaggle's server-side idle timer.
import IPython
js_code = '''
console.log("🛡️ Shakey Anti-Idle Registered.");
if (window._shakeyIdleTimer) clearInterval(window._shakeyIdleTimer);
window._shakeyIdleTimer = setInterval(function() {
    document.querySelector('body').click();
    console.log("💓 Shakey heartbeat: " + new Date().toISOString());
}, 55000);
'''
display(IPython.display.Javascript(js_code))

# ── 1. IMPORTS & KAGGLE SECRETS ───────────────────────────────────────────────
import os
import sys
import time
import queue
import signal
import shutil
import threading
import subprocess
import json
import traceback
from pathlib import Path
from datetime import datetime

# Install Hugging Face Hub silently before import
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import HfApi, login, snapshot_download  # Bug #4: module-level import
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

# ── Load all secrets with clear error messages ─────────────────────────────────
print("🔑 Loading Kaggle secrets...")
missing = []

try:
    NVIDIA_API_KEY = secrets.get_secret("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    print("  ✅ NVIDIA_API_KEY loaded")
except Exception as e:
    missing.append("NVIDIA_API_KEY")
    print(f"  ❌ NVIDIA_API_KEY MISSING — distillation will FAIL: {e}")

try:
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("  ✅ HF_TOKEN loaded")
except Exception as e:
    missing.append("HF_TOKEN")
    print(f"  ❌ HF_TOKEN MISSING — checkpoints will NOT be saved: {e}")
    HF_TOKEN = None

try:
    HF_REPO_ID = secrets.get_secret("HF_REPO_ID")
    os.environ["HF_REPO_ID"] = HF_REPO_ID
    print(f"  ✅ HF_REPO_ID loaded: {HF_REPO_ID}")
except Exception as e:
    missing.append("HF_REPO_ID")
    print(f"  ❌ HF_REPO_ID MISSING: {e}")
    HF_REPO_ID = None

if "NVIDIA_API_KEY" in missing:
    raise SystemExit("❌ FATAL: NVIDIA_API_KEY is required. Add it to Kaggle Secrets and restart.")

# ── 2. INSTALL RUST TOOLCHAIN ─────────────────────────────────────────────────
print("\n🦀 Installing Rust Toolchain (stable)...")
rustup_result = subprocess.run(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path",
    shell=True, capture_output=True, text=True
)
if rustup_result.returncode != 0:
    print(f"  Rustup stderr: {rustup_result.stderr[-500:]}")
    raise SystemExit("❌ FATAL: Rust installation failed.")

# Add cargo to PATH for this session
CARGO_BIN = os.path.expanduser("~/.cargo/bin")
os.environ["PATH"] = f"{CARGO_BIN}:{os.environ['PATH']}"

# Verify rustc is available
rust_version = subprocess.getoutput("rustc --version")
print(f"  ✅ Rust ready: {rust_version}")

# ── 3. CLONE / UPDATE SHAKEY REPOSITORY ──────────────────────────────────────
GITHUB_REPO = "myth-tools/shakey"  # username/repo — change if forked
IS_PRIVATE = False

repo_name = GITHUB_REPO.split("/")[-1]  # "shakey"
REPO_DIR = Path("/kaggle/working") / repo_name

if REPO_DIR.exists():
    print(f"\n✅ Repo found at {REPO_DIR}. Pulling latest code...")
    result = subprocess.run(["git", "-C", str(REPO_DIR), "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️ git pull warning: {result.stderr.strip()}")
else:
    print(f"\n🔄 Cloning {GITHUB_REPO}...")
    if IS_PRIVATE:
        try:
            gh_token = secrets.get_secret("GITHUB_TOKEN")
            repo_url = f"https://{gh_token}@github.com/{GITHUB_REPO}.git"
        except Exception:
            raise SystemExit("❌ FATAL: Private repo requires GITHUB_TOKEN in Kaggle Secrets.")
    else:
        repo_url = f"https://github.com/{GITHUB_REPO}.git"
    subprocess.check_call(["git", "clone", repo_url, str(REPO_DIR)])

# Change into repo directory — ALL paths below are relative to this
os.chdir(str(REPO_DIR))
print(f"📂 Working directory: {os.getcwd()}")

# ── Critical path setup ─────────────────────────────────────────────────────
# Bug #9 fix: SHAKEY_DATA_DIR is now honoured even in Kaggle mode (env.rs
# updated to check the env var before falling back to the hardcoded default).
# Setting it here makes the Rust binary respect a custom data path if needed.
KAGGLE_DATA_DIR = Path("/kaggle/working/shakey_data")
os.environ["SHAKEY_DATA_DIR"] = str(KAGGLE_DATA_DIR)

CHECKPOINT_DIR = KAGGLE_DATA_DIR / "checkpoints"
KNOWLEDGE_DIR  = KAGGLE_DATA_DIR / "knowledge"
LOGS_DIR       = KAGGLE_DATA_DIR / "logs"

for d in [CHECKPOINT_DIR, KNOWLEDGE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"  📁 Data dir     : {KAGGLE_DATA_DIR}")
print(f"  📁 Checkpoints  : {CHECKPOINT_DIR}")
print(f"  📁 Knowledge    : {KNOWLEDGE_DIR}")

# ── 4. HUGGING FACE CHECKPOINT SYNC ──────────────────────────────────────────
class ShakeyHfSync:
    """
    Async HF uploader and checkpoint restorer.
    - Restores latest checkpoint on startup (resumable training).
    - Watches checkpoint dir and uploads new step_* directories.
    - Uploads knowledge DB and configs on shutdown.
    - All uploads retry with exponential back-off (5 attempts).

    Upload queue items are 3-tuples:
        (kind, local_path, path_in_repo)
    where `path_in_repo` is the exact destination path on the HF repo.
    This makes routing explicit and eliminates the "all uploads go to
    checkpoints/" bug from v3.
    """
    def __init__(self):
        if not HF_TOKEN or not HF_REPO_ID:
            print("⚠️  HF sync disabled (missing token/repo).")
            self.enabled = False
            return
        self.enabled = True
        self.upload_queue = queue.Queue()
        self.stop_event   = threading.Event()
        # Bug #6 fix: track by step NAME (e.g. "step_42"), not absolute path,
        # so the check is stable regardless of how watch_path was specified.
        self.uploaded_ids = set()

        login(token=HF_TOKEN, add_to_git_credential=False)
        self.api    = HfApi()
        self.repo_id = HF_REPO_ID

        # Ensure HF repo exists (private)
        try:
            self.api.create_repo(repo_id=self.repo_id, private=True, exist_ok=True)
            print(f"  ✅ HF repo ready: {self.repo_id}")
        except Exception as e:
            print(f"  ⚠️ HF repo creation note: {e}")

        # Start background upload worker
        self.worker = threading.Thread(target=self._upload_worker, daemon=True, name="hf-uploader")
        self.worker.start()

    def restore_latest_checkpoint(self):
        """
        Download the most recent checkpoint from HF before training starts.
        This enables seamless resume after Kaggle session restart.
        """
        if not self.enabled:
            return
        try:
            print("\n🔄 Checking HF for existing checkpoints to restore...")
            repo_files = self.api.list_repo_files(repo_id=self.repo_id)
            # Find all files under checkpoints/step_*/
            step_dirs = set()
            for f in repo_files:
                parts = f.split("/")
                if len(parts) >= 2 and parts[0] == "checkpoints" and parts[1].startswith("step_"):
                    step_dirs.add(parts[1])

            if not step_dirs:
                print("  ℹ️  No prior checkpoints found on HF. Starting fresh.")
                return

            # Sort by step number descending, restore the latest
            def step_num(s):
                try: return int(s.split("_")[1])
                except: return 0
            latest = sorted(step_dirs, key=step_num, reverse=True)[0]
            local_ckpt = CHECKPOINT_DIR / latest

            if local_ckpt.exists():
                print(f"  ✅ Checkpoint {latest} already present locally. Skipping download.")
                # Bug #6 fix: mark by name so watcher won't re-upload it
                self.uploaded_ids.add(latest)
                return

            print(f"  ⬇️  Restoring checkpoint: {latest} from HF...")

            # Bug #4 fix: snapshot_download is a module-level function, NOT a
            # method of HfApi. Calling self.api.snapshot_download() raises
            # AttributeError. Also use "**" pattern for full recursive download.
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(CHECKPOINT_DIR),
                allow_patterns=f"checkpoints/{latest}/**",
                token=HF_TOKEN,
            )
            # Move from nested path to flat path if needed
            nested = CHECKPOINT_DIR / "checkpoints" / latest
            if nested.exists() and not local_ckpt.exists():
                shutil.move(str(nested), str(local_ckpt))
                # Clean up empty parent
                nested_parent = CHECKPOINT_DIR / "checkpoints"
                try:
                    nested_parent.rmdir()
                except OSError:
                    pass

            print(f"  ✅ Checkpoint {latest} restored to {local_ckpt}")
            # Bug #6 fix: mark by NAME so the watcher never re-uploads it
            self.uploaded_ids.add(latest)
        except Exception as e:
            print(f"  ⚠️ Checkpoint restore failed (starting fresh): {e}")

    def watch_directory(self, watch_path: str, interval: int = 90):
        """Watch for new step_* checkpoint dirs and queue them for upload."""
        if not self.enabled:
            return
        def _watcher():
            watch = Path(watch_path)
            while not self.stop_event.is_set():
                time.sleep(interval)
                if not watch.exists():
                    continue
                try:
                    for item in watch.iterdir():
                        # Bug #6 fix: key on item.name ("step_42"), not str(item)
                        # (absolute path), so it always matches self.uploaded_ids
                        # regardless of whether watch_path was relative or absolute.
                        if item.is_dir() and item.name.startswith("step_") \
                                and item.name not in self.uploaded_ids:
                            self.uploaded_ids.add(item.name)
                            # Bug #5 fix: explicit path_in_repo in the queue tuple
                            self.upload_queue.put(
                                ("dir", str(item), f"checkpoints/{item.name}")
                            )
                            print(f"\n👀 [HF Watcher] New checkpoint queued: {item.name}")
                except Exception as e:
                    print(f"  ⚠️ Watcher error: {e}")
        threading.Thread(target=_watcher, daemon=True, name="hf-watcher").start()
        print(f"  👁️  HF watcher active on: {watch_path} (interval={interval}s)")

    def sync_configs(self, interval: int = 300):
        """Upload configs/ dir every N seconds so agent.yaml changes persist."""
        if not self.enabled:
            return
        def _config_syncer():
            while not self.stop_event.is_set():
                time.sleep(interval)
                if not self.stop_event.is_set() and Path("configs").exists():
                    # Bug #5 fix: upload to "configs/" on HF, not "checkpoints/"
                    self.upload_queue.put(("dir", "configs", "configs"))
        threading.Thread(target=_config_syncer, daemon=True, name="hf-config-sync").start()

    def upload_vital_database(self, db_path: str):
        """Queue knowledge DB for upload (call on shutdown)."""
        if not self.enabled:
            return
        p = Path(db_path)
        if p.exists():
            # Bug #5 fix: upload to "database/" on HF, not "checkpoints/"
            self.upload_queue.put(("file", str(p), f"database/{p.name}"))
            print(f"  💾 Knowledge DB queued for upload: {p.name}")

    def _upload_worker(self):
        while not self.stop_event.is_set() or not self.upload_queue.empty():
            try:
                # Bug #5 fix: queue items are now 3-tuples with explicit path_in_repo
                kind, local_path, path_in_repo = self.upload_queue.get(timeout=2)
                self._upload_with_retry(kind, local_path, path_in_repo)
                self.upload_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"  ❌ Upload worker error: {e}")

    def _upload_with_retry(self, kind, local_path, path_in_repo, max_attempts=5):
        """
        Upload a file or folder to HF with exponential back-off retry.
        `path_in_repo` is the exact destination path on the HF repo,
        e.g. "checkpoints/step_42", "configs", "database/agent.redb".
        """
        for attempt in range(max_attempts):
            try:
                if kind == "dir":
                    print(f"  ☁️  Uploading folder → {path_in_repo} (attempt {attempt+1}/{max_attempts})...")
                    self.api.upload_folder(
                        folder_path=local_path,
                        repo_id=self.repo_id,
                        path_in_repo=path_in_repo,
                        ignore_patterns=["*.tmp", "*.lock"],
                    )
                else:
                    print(f"  ☁️  Uploading file → {path_in_repo} (attempt {attempt+1}/{max_attempts})...")
                    self.api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=path_in_repo,
                        repo_id=self.repo_id,
                    )
                print(f"  ✅ Uploaded: {path_in_repo}")
                return
            except Exception as e:
                wait = 2 ** attempt
                print(f"  ⚠️ Upload attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        print(f"  ❌ All {max_attempts} upload attempts failed for: {path_in_repo}")

    def finish(self):
        """Drain the upload queue and stop all threads gracefully."""
        print("\n⏳ Draining upload queue before shutdown...")
        # Bug #7 fix: set stop_event first so background threads stop enqueuing,
        # then use queue.join() which atomically waits for all task_done() calls.
        # The old "poll empty() then sleep" pattern had a race window where items
        # enqueued after the empty() check but before stop_event.set() were lost.
        self.stop_event.set()
        try:
            self.upload_queue.join()  # Blocks until every put() has a task_done()
        except Exception as e:
            print(f"  ⚠️ Queue drain warning: {e}")
        self.worker.join(timeout=60)
        print("  🎉 HF sync complete.")


# ── 5. COMPILE SHAKEY ─────────────────────────────────────────────────────────
print("\n🦀 Compiling Project Shakey (Release Mode)...")
print("   This takes ~4-8 minutes on first compile.")

# Set RUSTFLAGS for maximum CPU optimization (Kaggle has AVX2-capable CPUs)
os.environ["RUSTFLAGS"] = "-C target-cpu=native"

build_start = time.time()
try:
    build_result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "shakey-cli"],
        timeout=600,  # 10-minute build timeout
        capture_output=True,
        text=True,
    )
    build_elapsed = time.time() - build_start
    if build_result.returncode != 0:
        print(f"  ❌ Build FAILED after {build_elapsed:.0f}s")
        print("  STDERR (last 2000 chars):")
        print(build_result.stderr[-2000:])
        raise SystemExit("❌ FATAL: cargo build failed. Fix compilation errors and retry.")
    print(f"  ✅ Build complete in {build_elapsed:.0f}s")
except subprocess.TimeoutExpired:
    raise SystemExit("❌ FATAL: Build timed out after 10 minutes. Kaggle OOM or hung linker.")

BINARY = "./target/release/shakey-cli"
assert Path(BINARY).exists(), f"Binary not found at {BINARY} after successful build"

# ── 6. INIT HF SYNC + RESTORE CHECKPOINT ─────────────────────────────────────
hf_sync = ShakeyHfSync()

# CRITICAL: Restore previous checkpoint BEFORE starting training
# This makes training fully resumable across Kaggle session restarts
hf_sync.restore_latest_checkpoint()

# Start watchers AFTER restore to avoid re-uploading restored checkpoints
hf_sync.watch_directory(watch_path=str(CHECKPOINT_DIR), interval=90)
hf_sync.sync_configs(interval=300)

# ── 7. INITIALIZE MODEL (if no checkpoint found) ──────────────────────────────
init_needed = not any(CHECKPOINT_DIR.glob("step_*"))
if init_needed:
    print("\n🌱 No checkpoint found — initializing fresh Seed model...")
    # Bug #8 fix: removed the "--" separator between global flags and the
    # subcommand name. With Clap, "--" signals "end of options" which causes
    # "init"/"evolve" to be parsed as positional arguments rather than
    # subcommands, producing an argument parse error.
    init_result = subprocess.run(
        [BINARY, "--kaggle", "init", "--stage", "seed"],
        timeout=120,
        capture_output=True,
        text=True,
    )
    if init_result.returncode != 0:
        print(f"  ⚠️ Init stderr: {init_result.stderr[-500:]}")
        print("  Continuing anyway — evolve will create fresh weights...")
    else:
        print("  ✅ Seed model initialized.")
else:
    n_ckpts = len(list(CHECKPOINT_DIR.glob("step_*")))
    print(f"\n✅ Found {n_ckpts} existing checkpoint(s). Resuming training.")

# ── 8. LAUNCH AUTONOMOUS EVOLUTION LOOP ──────────────────────────────────────
print("\n🤖 Launching Sovereign OODA Evolution Loop...")
print("=" * 60)

# Bug #8 fix: "--" removed here too. The correct Clap invocation is:
#   shakey [GLOBAL_OPTIONS] <SUBCOMMAND> [SUBCOMMAND_OPTIONS]
evolve_cmd = [
    BINARY,
    "--kaggle",           # Enables Kaggle progress dashboard + optimizations
    "--log-level", "info",
    "evolve",
    "--max-cycles", "0",  # 0 = infinite (runs until Kaggle kills session)
]

process = None  # Bug #1 pre-init: ensures the name exists before the try block
try:
    # Launch with live output streaming (no buffering)
    evolve_env = os.environ.copy()
    evolve_env["RUST_LOG"] = "info"
    evolve_env["RUST_BACKTRACE"] = "1"

    process = subprocess.Popen(
        evolve_cmd,
        env=evolve_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout for single stream
        text=True,
        bufsize=1,  # Line-buffered
    )

    print(f"  🟢 Process PID: {process.pid}")
    print("  📊 Live logs below (progress dashboard updates every 60s):\n")

    # Stream output live so Kaggle doesn't think the cell is frozen.
    # This is ALSO the real anti-idle mechanism: continuous stdout activity
    # prevents Kaggle's server-side idle detection from firing.
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()
    exit_code = process.returncode

    if exit_code == 0:
        print("\n✅ Evolution loop completed cleanly.")
    elif exit_code < 0:
        # Bug #2 fix: handle all signal terminations generically.
        # process.returncode is -N for signal N. Using -signal.SIGTERM directly
        # risks enum/int comparison issues in some Python versions.
        sig_num = -exit_code
        try:
            sig_name = signal.Signals(sig_num).name
        except ValueError:
            sig_name = f"signal {sig_num}"
        print(f"\n⚠️  Process terminated by {sig_name}. Checkpoint should be saved.")
    else:
        print(f"\n⚠️  Process exited with code {exit_code}")

except KeyboardInterrupt:
    print("\n🛑 Ctrl+C detected — initiating graceful shutdown...")
    # Bug #1 fix: use the pre-initialised `process` variable instead of
    # "process" in dir() (which checks object attributes, not local variables).
    # Since process = None is set before the try block, this is always safe.
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

except subprocess.SubprocessError as e:
    print(f"\n❌ Subprocess error: {e}")
    traceback.print_exc()

finally:
    # ── 9. GRACEFUL SHUTDOWN: Upload everything before session ends ───────────
    print("\n" + "=" * 60)
    print("💾 SOVEREIGN SHUTDOWN: Securing all data to HuggingFace...")

    # Knowledge base (the agent's persistent memory)
    knowledge_db = KNOWLEDGE_DIR / "agent.redb"
    hf_sync.upload_vital_database(str(knowledge_db))

    # Telemetry dashboard if it exists
    # Bug #5 fix: explicit path_in_repo so it goes to "telemetry/" not "checkpoints/"
    telemetry = KAGGLE_DATA_DIR / "telemetry_dashboard.json"
    if telemetry.exists():
        hf_sync.upload_queue.put(
            ("file", str(telemetry), f"telemetry/{telemetry.name}")
        )

    # Configs (creator.yaml auto-patches need to persist)
    # Bug #5 fix: goes to "configs/" on HF, not "checkpoints/_configs_latest/"
    if Path("configs").exists():
        hf_sync.upload_queue.put(("dir", "configs", "configs"))

    # Drain queue and shut down
    hf_sync.finish()

    print("\n🎉 Session ended. All data secured.")
    # Bug #3 fix: HF_REPO_ID is None when the secret was missing.
    # Guard the URL print so we don't display "https://huggingface.co/None/..."
    if HF_REPO_ID:
        print(f"   Checkpoints : https://huggingface.co/{HF_REPO_ID}/tree/main/checkpoints")
        print(f"   Database    : https://huggingface.co/{HF_REPO_ID}/tree/main/database")
        print(f"   Configs     : https://huggingface.co/{HF_REPO_ID}/tree/main/configs")
        print(f"   Telemetry   : https://huggingface.co/{HF_REPO_ID}/tree/main/telemetry")
    else:
        print("   ⚠️  HF sync was disabled — no remote checkpoints or data were saved.")
```

---

### Quick Reference

| Secret | Value |
|--------|-------|
| `NVIDIA_API_KEY` | `nvapi-...` |
| `HF_TOKEN` | `hf_...` (Write access) |
| `HF_REPO_ID` | `Sishv/Shakey-Checkpoints` |

### HF Repo Layout (after first run)

| Path on HF | Contents |
|------------|----------|
| `checkpoints/step_N/` | Model weights + training state |
| `database/agent.redb` | Agent knowledge base |
| `configs/` | Agent YAML configs (auto-synced every 5 min) |
| `telemetry/` | Telemetry dashboard JSON |

### Key Improvements in v4

| # | Bug | Fix |
|---|-----|-----|
| 1 | `"process" in dir()` → NameError risk | Pre-initialise `process = None` before `try` |
| 2 | `-signal.SIGTERM` enum negation not portable | Handle all `exit_code < 0` generically via `signal.Signals` |
| 3 | `HF_REPO_ID = None` printed in `finally` URLs | Guard with `if HF_REPO_ID:` |
| 4 | `self.api.snapshot_download()` → AttributeError | Import and call module-level `snapshot_download()`; use `**` glob |
| 5 | All uploads go to `checkpoints/` prefix | Queue redesigned to 3-tuple `(kind, local, path_in_repo)` |
| 6 | `uploaded_ids` keyed on absolute path → fragile | Key on `item.name` (`"step_42"`) instead |
| 7 | `finish()` drain races between `empty()` and `set()` | Use `queue.join()` for atomic drain |
| 8 | `"--"` separator breaks Clap subcommand parsing | Removed `"--"` from `init` and `evolve` invocations |
| 9 | `SHAKEY_DATA_DIR` silently ignored in Kaggle mode | `env.rs` updated to check env var first in all environments |
| 10 | JS `body.click()` doesn't reset server-side idle | Documented; real keepalive is the live stdout stream |
