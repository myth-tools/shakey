use std::collections::HashMap;

/// Kali Pro Toolkit: A high-fidelity library of perfect Kali-Linux tool templates.
pub struct KaliProToolkit {
    templates: HashMap<String, String>,
}

impl Default for KaliProToolkit {
    fn default() -> Self {
        Self::new()
    }
}

impl KaliProToolkit {
    pub fn new() -> Self {
        let mut t = HashMap::new();
        t.insert("nmap_full".into(), "nmap -T4 -A -v -Pn {{target}}".into());
        t.insert(
            "gobuster_dir".into(),
            "gobuster dir -u {{target}} -w /usr/share/wordlists/dirb/common.txt -t 50".into(),
        );
        t.insert(
            "hydra_ssh".into(),
            "hydra -l root -P /usr/share/wordlists/rockyou.txt {{target}} ssh".into(),
        );
        t.insert("nikto_scan".into(), "nikto -h {{target}} -C all".into());
        t.insert(
            "msf_search".into(),
            "msfconsole -q -x 'search {{query}}; exit'".into(),
        );
        t.insert("curl_header".into(), "curl -I -L {{target}}".into());
        t.insert(
            "openssl_check".into(),
            "openssl s_client -connect {{target}}:443 -brief".into(),
        );

        Self { templates: t }
    }

    pub fn get_template(&self, key: &str, params: &HashMap<String, String>) -> Option<String> {
        let tmpl = self.templates.get(key)?;
        let mut filled = tmpl.clone();
        for (k, v) in params {
            filled = filled.replace(&format!("{{{{{}}}}}", k), v);
        }
        Some(filled)
    }

    pub fn list_tools(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }
}
