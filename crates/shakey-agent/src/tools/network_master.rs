use anyhow::Result;
use serde::Deserialize;
use std::process::Command;

#[derive(Debug, Deserialize)]
pub struct NetworkScanRequest {
    pub target: String,
    pub ports: Option<String>,
    pub aggressive: bool,
}

/// Network Master: Sovereign-Grade networking diagnostics and reconnaissance.
#[derive(Default)]
pub struct NetworkMaster;

impl NetworkMaster {
    pub fn new() -> Self {
        Self
    }

    pub fn scan(&self, target: &str, ports: Option<&str>, aggressive: bool) -> Result<String> {
        tracing::info!(target: "shakey::cyber", "📡 NetworkMaster: Scanning target '{}'", target);

        let mut nmap = Command::new("nmap");
        nmap.arg("-T4"); // Fast execution

        if aggressive {
            nmap.arg("-A"); // OS detection, version detection, script scanning, traceroute
        }

        if let Some(p) = ports {
            nmap.arg("-p").arg(p);
        }

        nmap.arg(target);

        let output = nmap.output()?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("Nmap Failed: {}", stderr))
        }
    }
}
