# 🚀 AutoPentest Assistant - RedTeam v7.1

**Complete penetration testing framework with Bloodhound Parser + ExploitDB Integration**  

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Kali-lightgrey)](https://www.kali.org/)
[![Tools](https://img.shields.io/badge/tools-50+-brightgreen)](README.md)

## 🎯 What's New in v7.1

- **Bloodhound Parser** - Ultra-fast AD attack path analysis with simdjson
- **ExploitDB Integration** - 50,000+ exploits with intelligent searching
- **Active Directory Modules** - Complete AD reconnaissance and vulnerability scanning
- **Tool Integration Framework** - 50+ security tools automated
- **Advanced Risk Scoring** - AI-powered risk assessment

## ✨ Features

### 🔍 **Intelligent Scanning**
- **Async Port Scanning** - 500+ concurrent connections
- **Service Fingerprinting** - Automatic service/version detection
- **Banner Analysis** - Advanced vulnerability detection from banners
- **Multi-target Support** - IP, CIDR, domains with auto-expansion

### 🛡️ **Active Directory**
- **Bloodhound Integration** - Parse and analyze AD attack paths
- **Domain Controller Detection** - Automatic AD environment mapping
- **Kerberoasting** - Automated service principal name attacks
- **DCSync Detection** - Identify privilege escalation paths
- **Zerologon/EternalBlue** - Automatic vulnerability checking

### 💥 **Exploitation**
- **ExploitDB Integration** - 50,000+ exploits with smart search
- **Metasploit Automation** - RC file generation and module execution
- **CVSS Scoring** - Vulnerability prioritization
- **Exploit Risk Assessment** - Intelligent exploit selection

### 🛠️ **Tool Integration**
# 50+ Integrated Tools:
- NMAP, Nuclei, Nikto, SQLMap
- CrackMapExec, BloodHound, Impacket
- Gobuster, Dirb, WPScan, Hydra
- John, Hashcat, Metasploit
- SearchSploit, Ldapsearch, Enum4linux

### Installation
# Clone and install
git clone https://github.com/your-org/autopentest-ultimate.git
cd autopentest-ultimate

# Install dependencies
pip install aiohttp beautifulsoup4 requests pyyaml simdjson networkx

# Make executable
chmod +x autopentest_ultimate.py
### Basic Usage

# Comprehensive network scan
./autopentest_ultimate.py 192.168.1.0/24

# AD environment scanning
./autopentest_ultimate.py 10.0.0.1-10.0.0.50 --ad-scan --run-tools

# Bloodhound analysis
./autopentest_ultimate.py --bloodhound bloodhound_data.zip

# ExploitDB search
./autopentest_ultimate.py --search-exploits "Apache 2.4.49"


## 📋 Usage Examples

### 1. Complete Network Assessment

# Full scan with all tools
./autopentest_ultimate.py 192.168.1.1/24 --run-tools --output-dir scan_results

# Specific port range with AD focus
./autopentest_ultimate.py 10.0.0.1-10.0.0.100 -p 1-1000 --ad-scan

# Web application testing
./autopentest_ultimate.py webapp.com -p 80,443,8080,8443 --run-tools


### 2. Active Directory Attacks
```bash
# AD reconnaissance with credentials
./autopentest_ultimate.py dc.company.com --ad-scan --username admin --password Pass123 --domain COMPANY

# Bloodhound data collection and analysis
./autopentest_ultimate.py --bloodhound collected_data.zip --bloodhound-target "Domain Admins"

# Kerberoasting attack
./autopentest_ultimate.py dc.company.com --username user --password pass --domain COMPANY --run-tools
```

### 3. Exploit Management
```bash
# Search for exploits
./autopentest_ultimate.py --search-exploits "WordPress 5.1.1"
./autopentest_ultimate.py --search-exploits "Apache Tomcat" --service http --port 8080

# Download specific exploit
./autopentest_ultimate.py --download-exploit 49757

# Generate exploit suggestions for target
./autopentest_ultimate.py target.com --run-tools
```

## 🎯 Command Line Options

| Option | Description |
|--------|-------------|
| `targets` | IP/CIDR/domain to scan |
| `-p, --ports` | Ports to scan (default: common ports) |
| `--ad-scan` | Enable Active Directory scanning |
| `--run-tools` | Execute all tool integrations |
| `--bloodhound` | Analyze Bloodhound ZIP file |
| `--bloodhound-target` | Target group for path analysis |
| `--search-exploits` | Search ExploitDB |
| `--download-exploit` | Download exploit by ID |
| `--username` | Username for credentialed scans |
| `--password` | Password for credentialed scans |
| `--domain` | Domain for AD scans |
| `--output-dir` | Output directory for results |

## 🔧 Advanced Features

### Bloodhound Parser
```python
# Ultra-fast parsing with simdjson
parser = BloodhoundParser()
data = parser.parse_bloodhound_data("bloodhound.zip")

# Find attack paths to Domain Admins
paths = parser.find_attack_paths(target_node="Domain Admins")

# Generate attack plan
attack_plan = parser.generate_attack_plan(paths)
```

### ExploitDB Integration
```python
# Search exploits
exploitdb = ExploitDBIntegration()
exploits = exploitdb.search_exploits("Apache 2.4.49", "http", 80)

# Risk-based sorting
exploits.sort(key=lambda x: x.risk_level.value, reverse=True)

# Download exploit code
exploit_code = exploitdb.get_exploit_code("49757")
```

### Tool Automation
```python
# Run integrated tools
runner = ToolRunner()
result = await runner.run_tool(tool_config, target, credentials)

# Generate tool commands based on scan results
commands = runner.generate_tool_commands(scan_result)
```

## 📊 Output & Reporting

### Sample Output
```
🚀 AutoPentest Assistant ULTIMATE v7.1
[INFO] Starting comprehensive scan for 5 targets
[SUCCESS] Found 3 hosts with open ports
[INFO] AD Domain Controller detected: 192.168.1.10
[SUCCESS] Bloodhound analysis complete: 15 attack paths found
[INFO] Top 3 attack paths to Domain Admins:
  1. Risk: 9.2 - User -> Computer -> Group -> Domain Admins
  2. Risk: 8.7 - Computer -> User -> Group -> Domain Admins
  3. Risk: 7.9 - User -> Group -> Domain Admins
```

### Generated Files
```
results/
├── scan_report.json          # Complete scan results
├── bloodhound_analysis.json  # Attack path analysis
├── exploits/                 # Downloaded exploits
├── tool_outputs/            # Individual tool results
└── metasploit_commands.rc   # Automated Metasploit scripts
```

## 🏗 Architecture

### Core Components
```
AutoPentest Assistant/
├── Scanner Manager          # Orchestrates all scanning
├── Async Port Scanner       # High-performance port scanning
├── Bloodhound Parser        # AD attack path analysis (simdjson)
├── ExploitDB Integration    # 50,000+ exploit database
├── Tool Integration Manager # 50+ security tools
├── AD Scanner              # Active Directory reconnaissance
├── Vulnerability Scanner   # CVE and risk assessment
└── Report Generator        # Multi-format output
```

### Tool Integration Framework
```python
TOOL_INTEGRATIONS = {
    "nmap_tcp_full": ToolIntegration(...),
    "nuclei_scan": ToolIntegration(...),
    "crackmapexec": ToolIntegration(...),
    "bloodhound_collect": ToolIntegration(...),
    "searchsploit": ToolIntegration(...),
    # ... 45+ more tools
}
```

## 🔒 Security Considerations

⚠️ **Legal & Ethical Usage**

This tool is designed for:
- Authorized penetration testing
- Security research and education
- Vulnerability assessment in owned environments
- Red team exercises with proper authorization

**Always obtain proper authorization before scanning any network or system.**

## 🐛 Troubleshooting

### Common Issues
```bash
# Bloodhound parsing slow
pip install simdjson

# Missing tools
sudo apt install nmap nikto sqlmap gobuster

# Network timeouts
./autopentest_ultimate.py target.com --ports 80,443,22
```

### Performance Tips
- Use specific port ranges for large networks
- Limit concurrent connections with `--max-hosts`
- Use credentialed scans for comprehensive AD assessment
- Process Bloodhound data separately for large environments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/your-org/autopentest-ultimate.git
cd autopentest-ultimate
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Bloodhound** - AD attack path mapping
- **ExploitDB** - Offensive Security exploit database
- **Metasploit** - Penetration testing framework
- **NMAP** - Network discovery and security auditing
- **SimdJSON** - Ultra-fast JSON parsing

---

**Disclaimer**: This tool should only be used on systems you own or have explicit permission to test. Unauthorized scanning and exploitation is illegal.

**Stay Ethical. Stay Secure. 🛡️**
