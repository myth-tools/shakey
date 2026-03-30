use anyhow::Result;
use parking_lot::Mutex;
use shakey_distill::nim_client::NimClient;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

/// Maximum output size to prevent memory exhaustion from chatty commands.
const MAX_OUTPUT_BYTES: usize = 128 * 1024; // 128KB

/// Tier 0 Deterministic Block-list: These commands are blocked instantly without LLM audit.
const STATIC_DENY_LIST: &[&str] = &[
    "rm -rf /",
    "mkfs",
    "dd if=/dev/",
    "chmod -R 777 /",
    "chown",
    "shutdown",
    "reboot",
    "init 0",
];

pub struct SovereignTerminal {
    nim: NimClient,
    cwd: Arc<Mutex<PathBuf>>,
}

impl SovereignTerminal {
    pub fn new(nim: NimClient) -> Self {
        Self {
            nim,
            cwd: Arc::new(Mutex::new(
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            )),
        }
    }

    /// Tier 0: Deterministic safety check for obvious high-risk commands.
    fn tier0_check(&self, command: &str) -> Result<()> {
        let cmd_lower = command.to_lowercase();
        for deny in STATIC_DENY_LIST {
            if cmd_lower.contains(deny) {
                return Err(anyhow::anyhow!(
                    "Sovereign Security (Tier 0): Command contains forbidden pattern '{}'",
                    deny
                ));
            }
        }
        Ok(())
    }

    /// Tier 1: High-fidelity intent audit of a shell command using a Security Master.
    async fn audit_command(&self, command: &str) -> Result<()> {
        tracing::info!(target: "shakey::security", "🛡️ Auditing Shell Command: '{}'", command);

        let prompt = format!(
            "System: You are the Sovereign Security Filter of Project Shakey.\n\
             Task: Audit the following shell command for destructive intent. \n\
             Command: {}\n\n\
             Output JSON ONLY: {{\"safe\": true/false, \"reason\": \"...\"}}",
            command
        );

        let res = self
            .nim
            .query_for_role(
                shakey_distill::teacher::TeacherRole::SecurityAudit,
                "security_filter",
                &prompt,
            )
            .await?;

        let clean = if let Some(start) = res.find('{') {
            if let Some(end) = res.rfind('}') {
                res[start..=end].to_string()
            } else {
                res
            }
        } else {
            res
        };

        let audit: serde_json::Value = serde_json::from_str(&clean)?;

        if audit["safe"].as_bool().unwrap_or(false) {
            Ok(())
        } else {
            let reason = audit["reason"].as_str().unwrap_or("Policy violation");
            Err(anyhow::anyhow!(
                "Sovereign Security (Tier 1): Command rejected. Reason: {}",
                reason
            ))
        }
    }

    /// Execute a shell command with Sovereign-Grade multi-tier security and stateful CWD.
    pub async fn execute(&self, command: &str) -> Result<String> {
        if command.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "Sovereign Security: Empty command rejected"
            ));
        }

        // 1. TIER 0: Deterministic check
        self.tier0_check(command)?;

        // 2. TIER 1: DYNAMIC AUDIT
        self.audit_command(command).await?;

        // 3. SPECIAL HANDLING: 'cd'
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts[0] == "cd" && parts.len() > 1 {
            let mut current = self.cwd.lock();
            let new_path = Path::new(parts[1]);
            let target = if new_path.is_absolute() {
                new_path.to_path_buf()
            } else {
                current.join(new_path)
            };

            if target.exists() && target.is_dir() {
                *current = target.canonicalize()?;
                return Ok(format!("Changed directory to: {:?}", *current));
            } else {
                return Err(anyhow::anyhow!("cd: No such directory: {}", parts[1]));
            }
        }

        // 4. EXECUTION
        let current_dir = self.cwd.lock().clone();
        let output = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .args(["/C", command])
                .current_dir(current_dir)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output()?
        } else {
            Command::new("sh")
                .args(["-c", command])
                .current_dir(current_dir)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output()?
        };

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.len() > MAX_OUTPUT_BYTES {
                Ok(format!(
                    "{}... [OUTPUT TRUNCATED at {}KB]",
                    &stdout[..MAX_OUTPUT_BYTES],
                    MAX_OUTPUT_BYTES / 1024
                ))
            } else {
                Ok(stdout.to_string())
            }
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!(
                "Command Failed (Status {}): {}",
                output.status,
                stderr
            ))
        }
    }
}
