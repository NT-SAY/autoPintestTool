#!/usr/bin/env python3
"""
🚀 AutoPentest Assistant - ULTIMATE WEAPON v7.1
Complete penetration testing framework with ALL features + Bloodhound Parser + ExploitDB Integration
"""

import asyncio
import aiohttp
import json
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
import re
import ipaddress
import ssl
import concurrent.futures
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from enum import Enum, auto
from urllib.parse import urljoin, urlparse, quote
import time
import argparse
import hashlib
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random
import string
import os
from pathlib import Path
import csv
import html
import dataclasses
from datetime import datetime, timedelta
import logging
import uuid
import hmac
import binascii
import tempfile
import zipfile
import io
import yaml
import requests
from bs4 import BeautifulSoup

# ==================== BLOODHOUND PARSER DEPENDENCIES ====================

try:
    import simdjson
    SIMDJSON_AVAILABLE = True
except ImportError:
    simdjson = None
    SIMDJSON_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

# ==================== ENUM CLASSES (ДОЛЖНЫ БЫТЬ ПЕРВЫМИ!) ====================

class PortStatus(Enum):
    OPEN = "open"
    CLOSED = "closed" 
    FILTERED = "filtered"
    UNFILTERED = "unfiltered"
    OPEN_FILTERED = "open|filtered"
    CLOSED_FILTERED = "closed|filtered"

class RiskLevel(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3 
    LOW = 2
    INFO = 1

class ServiceType(Enum):
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    FTP = "ftp"
    SMTP = "smtp"
    TELNET = "telnet"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    RDP = "rdp"
    VNC = "vnc"
    SNMP = "snmp"
    DNS = "dns"
    SMB = "smb"
    LDAP = "ldap"
    KERBEROS = "kerberos"
    WINRM = "winrm"
    UNKNOWN = "unknown"

class ADServiceType(Enum):
    DOMAIN_CONTROLLER = "domain_controller"
    MEMBER_SERVER = "member_server" 
    WORKSTATION = "workstation"
    SQL_SERVER = "sql_server"
    EXCHANGE = "exchange"
    ADFS = "adfs"
    CERTIFICATE_AUTHORITY = "certificate_authority"

class AttackFramework(Enum):
    METASPLOIT = "metasploit"
    NUCLEI = "nuclei"
    BURP_SUITE = "burp_suite"
    SQLMAP = "sqlmap"
    JOHN = "john"
    HASHCAT = "hashcat"
    CRACKMAPEXEC = "crackmapexec"
    BLOODHOUND = "bloodhound"
    IMPACKET = "impacket"
    NMAP = "nmap"
    GOBUSTER = "gobuster"
    DIRB = "dirb"
    NIKTO = "nikto"
    WPSCAN = "wpscan"
    HYDRA = "hydra"
    EXPLOITDB = "exploitdb"

# ==================== DATACLASSES (ПОСЛЕ ENUM!) ====================

@dataclass
class PortResult:
    port: int
    status: PortStatus
    service: str = ""
    version: str = ""
    banner: str = ""
    cpe: str = ""
    protocol: str = "tcp"

@dataclass
class HostResult:
    ip: str
    hostname: str = ""
    os: str = ""
    mac: str = ""
    ports: List[PortResult] = field(default_factory=list)
    services: Dict[str, Any] = field(default_factory=dict)
    vulnerabilities: List[Any] = field(default_factory=list)
    exploits: List[Any] = field(default_factory=list)
    risk_score: float = 0.0
    is_windows: bool = False
    is_domain_controller: bool = False

@dataclass
class ADHostResult:
    ip: str
    hostname: str = ""
    domain: str = ""
    os_version: str = ""
    ad_roles: List[str] = field(default_factory=list)
    shares: List[Dict] = field(default_factory=list)
    users: List[Dict] = field(default_factory=list)
    groups: List[Dict] = field(default_factory=list)
    gpo: List[Dict] = field(default_factory=list)
    sessions: List[Dict] = field(default_factory=list)
    logged_on_users: List[Dict] = field(default_factory=list)
    trust_relationships: List[Dict] = field(default_factory=list)
    is_domain_controller: bool = False
    domain_sid: str = ""
    forest: str = ""

@dataclass
class Vulnerability:
    id: str
    name: str
    description: str
    risk_level: RiskLevel
    cvss_score: float = 0.0
    cvss_vector: str = ""
    cve_id: str = ""
    service: str = ""
    port: int = 0
    exploit_available: bool = False
    exploit_db_id: str = ""
    metasploit_module: str = ""
    nuclei_template: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)

@dataclass
class Exploit:
    id: str
    title: str
    description: str
    author: str
    date: str
    type: str
    platform: str
    port: int
    cve_id: str = ""
    verified: bool = False
    download_url: str = ""
    code: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    service: str = ""
    language: str = ""
    path: str = ""

@dataclass
class ScanResult:
    targets: List[str]
    hosts: List[HostResult]
    start_time: datetime
    end_time: datetime
    ad_hosts: List[ADHostResult] = field(default_factory=list)
    scan_config: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    bloodhound_paths: List[Dict] = field(default_factory=list)
    exploits_found: List[Exploit] = field(default_factory=list)

@dataclass
class CVEData:
    cve_id: str
    description: str
    cvss_score: float
    cvss_vector: str
    published_date: str
    last_modified: str
    references: List[str]
    vulnerable_products: List[str]

@dataclass(frozen=True)
class ToolIntegration:
    name: str
    framework: AttackFramework
    command: str
    arguments: List[str] = field(default_factory=list)
    output_file: str = ""
    enabled: bool = True
    requires_credentials: bool = False

    def __hash__(self):
        return hash((self.name, self.framework.value, self.command, tuple(self.arguments)))

    def __eq__(self, other):
        if not isinstance(other, ToolIntegration):
            return False
        return (self.name == other.name and 
                self.framework == other.framework and
                self.command == other.command and
                self.arguments == other.arguments)


@dataclass
class BloodhoundPath:
    source: str
    target: str
    path: List[str]
    relationship_types: List[str]
    path_length: int
    risk_score: float
    description: str


# ==================== EXPLOITDB INTEGRATION ====================

class ExploitDBIntegration:
    def __init__(self, exploitdb_path: str = "/usr/share/exploitdb"):
        self.exploitdb_path = Path(exploitdb_path)
        self.csv_path = self.exploitdb_path / "files_exploits.csv"
        self.exploits_path = self.exploitdb_path / "exploits"
        self.logger = Logger()
        
        # Cache for fast searching
        self.exploits_cache = []
        self._load_exploits_cache()
    
    def _load_exploits_cache(self):
        """Load exploits cache from CSV"""
        if not self.csv_path.exists():
            self.logger.warning(f"ExploitDB not found at {self.csv_path}")
            return
        
        try:
            with open(self.csv_path, 'r', encoding='latin-1') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.exploits_cache.append(row)
            
            self.logger.success(f"Loaded {len(self.exploits_cache)} exploits from ExploitDB")
        except Exception as e:
            self.logger.error(f"Failed to load ExploitDB: {e}")
    
    def search_exploits(self, query: str, service: str = "", port: int = 0, platform: str = "") -> List[Exploit]:
        """Search exploits by various criteria"""
        exploits = []
        
        if not self.exploits_cache:
            self.logger.warning("ExploitDB cache not loaded")
            return exploits
        
        query_lower = query.lower()
        service_lower = service.lower() if service else ""
        platform_lower = platform.lower() if platform else ""
        
        for exploit_data in self.exploits_cache:
            if self._matches_search(exploit_data, query_lower, service_lower, port, platform_lower):
                exploit = self._create_exploit_object(exploit_data)
                if exploit:
                    exploits.append(exploit)
        
        # Sort by relevance
        exploits.sort(key=lambda x: (
            x.risk_level.value, 
            1 if query_lower in x.title.lower() else 0,
            1 if service_lower in x.title.lower() else 0
        ), reverse=True)
        
        self.logger.info(f"Found {len(exploits)} exploits for query: {query}")
        return exploits[:50]  # Limit results
    
    def _matches_search(self, exploit_data: Dict, query: str, service: str, port: int, platform: str) -> bool:
        """Check if exploit matches search criteria"""
        description = exploit_data.get('description', '').lower()
        
        # Basic query search
        if query and query not in description:
            return False
        
        # Service search
        if service and service not in description:
            return False
            
        # Platform search
        if platform and platform not in description:
            return False
            
        # Port search (if specified)
        if port > 0:
            port_str = str(port)
            if port_str not in description:
                return False
        
        return True
    
    def _create_exploit_object(self, exploit_data: Dict) -> Optional[Exploit]:
        """Create Exploit object from CSV data"""
        try:
            # Determine risk based on description
            risk_level = self._determine_exploit_risk(exploit_data.get('description', ''))
            
            # Determine service
            service = self._determine_exploit_service(exploit_data.get('description', ''))
            
            # Determine language
            language = self._determine_exploit_language(exploit_data.get('file', ''))
            
            exploit = Exploit(
                id=exploit_data.get('id', ''),
                title=exploit_data.get('description', 'No title'),
                description=exploit_data.get('description', 'No description'),
                author=exploit_data.get('author', 'Unknown'),
                date=exploit_data.get('date', ''),
                type=exploit_data.get('type', 'unknown'),
                platform=exploit_data.get('platform', 'multiple'),
                port=0,  # Will be determined later
                risk_level=risk_level,
                service=service,
                language=language,
                path=exploit_data.get('file', '')
            )
            
            # Extract CVE from description
            cve_match = re.search(r'CVE-\d{4}-\d+', exploit_data.get('description', ''))
            if cve_match:
                exploit.cve_id = cve_match.group(0)
            
            # Determine port from description
            port_match = re.search(r'port\s+(\d+)', exploit_data.get('description', ''), re.IGNORECASE)
            if port_match:
                exploit.port = int(port_match.group(1))
            
            return exploit
            
        except Exception as e:
            self.logger.warning(f"Failed to create exploit object: {e}")
            return None
    
    def _determine_exploit_risk(self, description: str) -> RiskLevel:
        """Determine exploit risk level based on description"""
        description_lower = description.lower()
        
        high_risk_indicators = [
            'remote', 'rce', 'code execution', 'privilege escalation', 
            'root', 'admin', 'arbitrary code', 'buffer overflow'
        ]
        
        medium_risk_indicators = [
            'dos', 'denial of service', 'crash', 'memory leak',
            'information disclosure', 'xss', 'cross site'
        ]
        
        low_risk_indicators = [
            'bypass', 'authentication', 'authorization', 'weak'
        ]
        
        if any(indicator in description_lower for indicator in high_risk_indicators):
            return RiskLevel.HIGH
        elif any(indicator in description_lower for indicator in medium_risk_indicators):
            return RiskLevel.MEDIUM
        elif any(indicator in description_lower for indicator in low_risk_indicators):
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO
    
    def _determine_exploit_service(self, description: str) -> str:
        """Determine exploit service based on description"""
        description_lower = description.lower()
        
        service_mapping = {
            'http': ['http', 'web', 'apache', 'nginx', 'iis'],
            'ssh': ['ssh'],
            'ftp': ['ftp'],
            'smb': ['smb', 'samba'],
            'rdp': ['rdp', 'remote desktop'],
            'mysql': ['mysql', 'mariadb'],
            'postgresql': ['postgres', 'postgresql'],
            'dns': ['dns', 'bind'],
            'smtp': ['smtp', 'mail'],
            'telnet': ['telnet']
        }
        
        for service, keywords in service_mapping.items():
            if any(keyword in description_lower for keyword in keywords):
                return service
        
        return "unknown"
    
    def _determine_exploit_language(self, file_path: str) -> str:
        """Determine exploit language based on file extension"""
        extension_map = {
            '.py': 'python',
            '.rb': 'ruby',
            '.c': 'c',
            '.cpp': 'c++',
            '.php': 'php',
            '.pl': 'perl',
            '.java': 'java',
            '.sh': 'bash',
            '.txt': 'text'
        }
        
        for ext, lang in extension_map.items():
            if file_path.endswith(ext):
                return lang
        
        return "unknown"
    
    def get_exploit_code(self, exploit_id: str) -> Optional[str]:
        """Get exploit code by ID"""
        try:
            # Find exploit in cache
            exploit_data = next((e for e in self.exploits_cache if e.get('id') == exploit_id), None)
            if not exploit_data:
                return None
            
            file_path = exploit_data.get('file')
            if not file_path:
                return None
            
            full_path = self.exploits_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                self.logger.warning(f"Exploit file not found: {full_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get exploit code: {e}")
            return None
    
    def search_exploits_by_cve(self, cve_id: str) -> List[Exploit]:
        """Search exploits by CVE ID"""
        return self.search_exploits(cve_id)
    
    def search_exploits_by_service_and_version(self, service: str, version: str) -> List[Exploit]:
        """Search exploits by service and version"""
        query = f"{service} {version}"
        return self.search_exploits(query, service)
    
    def generate_exploit_suggestions(self, host: HostResult) -> List[Exploit]:
        """Generate exploit suggestions based on host information"""
        exploits = []
        
        # Search by services and versions
        for port in host.ports:
            if port.status == PortStatus.OPEN:
                # Search by service
                service_exploits = self.search_exploits(port.service, port.service, port.port)
                exploits.extend(service_exploits)
                
                # Search by version
                if port.version:
                    version_exploits = self.search_exploits_by_service_and_version(port.service, port.version)
                    exploits.extend(version_exploits)
        
        # Search by OS
        if host.os:
            os_exploits = self.search_exploits(host.os, platform=host.os)
            exploits.extend(os_exploits)
        
        # Remove duplicates and sort by risk
        unique_exploits = []
        seen_ids = set()
        
        for exploit in exploits:
            if exploit.id not in seen_ids:
                seen_ids.add(exploit.id)
                unique_exploits.append(exploit)
        
        unique_exploits.sort(key=lambda x: x.risk_level.value, reverse=True)
        
        return unique_exploits[:20]  # Limit quantity
    
    def download_exploit(self, exploit_id: str, output_dir: str = "exploits") -> Optional[str]:
        """Download exploit to specified directory"""
        try:
            exploit_code = self.get_exploit_code(exploit_id)
            if not exploit_code:
                return None
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = Path(output_dir) / f"exploit_{exploit_id}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(exploit_code)
            
            self.logger.success(f"Exploit {exploit_id} saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download exploit: {e}")
            return None

# ==================== BLOODHOUND PARSER WITH SIMDJSON ====================

class BloodhoundParser:
    def __init__(self):
        self.parser = simdjson.Parser() if SIMDJSON_AVAILABLE else None
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.logger = Logger()
        
    def parse_bloodhound_data(self, zip_path: str) -> Dict[str, Any]:
        """Parse Bloodhound data from ZIP archive with maximum performance"""
        parsed_data = {
            'users': [],
            'groups': [],
            'computers': [],
            'domains': [],
            'sessions': [],
            'acls': [],
            'relationships': []
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract and parse each JSON file
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.json'):
                        with zip_ref.open(file_info.filename) as file:
                            content = file.read()
                            
                            if SIMDJSON_AVAILABLE:
                                # Ultra-fast parsing with simdjson
                                parsed = self.parser.parse(content)
                                data = parsed.as_dict()
                            else:
                                # Standard parsing
                                data = json.loads(content.decode('utf-8'))
                            
                            # Classify data by types
                            self._classify_bloodhound_data(data, parsed_data)
            
            # Build graph if networkx is available
            if NETWORKX_AVAILABLE:
                self._build_attack_graph(parsed_data)
            
            self.logger.success(f"Bloodhound data parsed: {len(parsed_data['users'])} users, {len(parsed_data['computers'])} computers")
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"Bloodhound parsing error: {e}")
            return parsed_data
    
    def _classify_bloodhound_data(self, data: Any, parsed_data: Dict[str, Any]):
        """Classify Bloodhound data by types"""
        if isinstance(data, list):
            for item in data:
                self._classify_single_item(item, parsed_data)
        elif isinstance(data, dict):
            self._classify_single_item(data, parsed_data)
    
    def _classify_single_item(self, item: Dict, parsed_data: Dict[str, Any]):
        """Classify single Bloodhound object"""
        object_type = item.get('ObjectType')
        
        if object_type == 'User':
            parsed_data['users'].append(item)
        elif object_type == 'Group':
            parsed_data['groups'].append(item)
        elif object_type == 'Computer':
            parsed_data['computers'].append(item)
        elif object_type == 'Domain':
            parsed_data['domains'].append(item)
        
        # Extract relationships
        if 'Aces' in item:
            parsed_data['acls'].extend(item['Aces'])
        
        # Extract sessions
        if 'Sessions' in item:
            parsed_data['sessions'].extend(item['Sessions'])
    
    def _build_attack_graph(self, parsed_data: Dict[str, Any]):
        """Build attack graph from Bloodhound data"""
        if not NETWORKX_AVAILABLE:
            return
            
        # Clear graph
        self.graph.clear()
        
        # Add nodes
        for user in parsed_data['users']:
            self.graph.add_node(user.get('ObjectIdentifier', ''), type='user', data=user)
        
        for computer in parsed_data['computers']:
            self.graph.add_node(computer.get('ObjectIdentifier', ''), type='computer', data=computer)
        
        for group in parsed_data['groups']:
            self.graph.add_node(group.get('ObjectIdentifier', ''), type='group', data=group)
        
        for domain in parsed_data['domains']:
            self.graph.add_node(domain.get('ObjectIdentifier', ''), type='domain', data=domain)
        
        # Add ACL relationships
        for acl in parsed_data['acls']:
            if 'PrincipalSID' in acl and 'ObjectIdentifier' in acl:
                self.graph.add_edge(
                    acl['PrincipalSID'],
                    acl['ObjectIdentifier'],
                    relationship='ACL',
                    rights=acl.get('RightName', '')
                )
        
        # Add group membership
        for group in parsed_data['groups']:
            group_sid = group.get('ObjectIdentifier', '')
            for member in group.get('Members', []):
                self.graph.add_edge(member.get('ObjectIdentifier', ''), group_sid, relationship='MemberOf')
        
        self.logger.success(f"Attack graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def find_attack_paths(self, start_node: str = None, target_node: str = "Domain Admins", max_paths: int = 10) -> List[BloodhoundPath]:
        """Find attack paths to target group"""
        if not NETWORKX_AVAILABLE or not self.graph:
            return []
        
        paths = []
        
        try:
            # Find target group
            target_group = None
            for node in self.graph.nodes(data=True):
                if target_node.lower() in node[0].lower() or (node[1].get('data') and target_node.lower() in node[1]['data'].get('Name', '').lower()):
                    target_group = node[0]
                    break
            
            if not target_group:
                self.logger.warning(f"Target group {target_node} not found")
                return []
            
            # If start node not specified, use all computers
            if not start_node:
                start_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('type') == 'computer']
            else:
                start_nodes = [start_node]
            
            for start in start_nodes:
                if start not in self.graph:
                    continue
                    
                try:
                    # Find all paths
                    all_paths = list(nx.all_simple_paths(self.graph, start, target_group, cutoff=6))
                    
                    for path in all_paths[:max_paths]:
                        bh_path = self._create_bloodhound_path(path)
                        if bh_path:
                            paths.append(bh_path)
                            
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    self.logger.warning(f"Path finding error for {start}: {e}")
            
            # Sort by risk
            paths.sort(key=lambda x: x.risk_score, reverse=True)
            
            self.logger.success(f"Found {len(paths)} attack paths to {target_node}")
            return paths
            
        except Exception as e:
            self.logger.error(f"Attack path finding error: {e}")
            return []
    
    def _create_bloodhound_path(self, path: List[str]) -> Optional[BloodhoundPath]:
        """Create BloodhoundPath object from path"""
        try:
            relationship_types = []
            path_description = []
            
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    rel_type = edge_data.get('relationship', 'Unknown')
                    relationship_types.append(rel_type)
                    
                    # Step description
                    source_node = self.graph.nodes[path[i]]
                    target_node = self.graph.nodes[path[i + 1]]
                    
                    step_desc = f"{source_node.get('type', 'Unknown')} -> {target_node.get('type', 'Unknown')} ({rel_type})"
                    path_description.append(step_desc)
            
            # Path risk calculation
            risk_score = self._calculate_path_risk(path, relationship_types)
            
            return BloodhoundPath(
                source=path[0],
                target=path[-1],
                path=path,
                relationship_types=relationship_types,
                path_length=len(path),
                risk_score=risk_score,
                description=" -> ".join(path_description)
            )
            
        except Exception as e:
            self.logger.warning(f"Path creation error: {e}")
            return None
    
    def _calculate_path_risk(self, path: List[str], relationships: List[str]) -> float:
        """Calculate attack path risk"""
        risk_score = 0.0
        
        # Base risk by path length
        base_risk = max(0, 10 - len(path))  # Shorter paths = higher risk
        
        # Risk by relationship types
        relationship_risks = {
            'AdminTo': 3.0,
            'HasSession': 2.0,
            'MemberOf': 2.0,
            'ACL': 1.5,
            'GenericAll': 3.0,
            'GenericWrite': 2.0,
            'WriteOwner': 2.5,
            'WriteDACL': 2.5
        }
        
        for rel in relationships:
            risk_score += relationship_risks.get(rel, 1.0)
        
        # Additional risk factors
        for node_id in path:
            node_data = self.graph.nodes.get(node_id, {})
            if node_data.get('type') == 'user':
                user_data = node_data.get('data', {})
                if user_data.get('Enabled') and not user_data.get('PasswordNeverExpires'):
                    risk_score += 0.5
        
        return min(risk_score + base_risk, 10.0)
    
    def generate_attack_plan(self, paths: List[BloodhoundPath]) -> Dict[str, Any]:
        """Generate attack plan based on found paths"""
        if not paths:
            return {}
        
        attack_plan = {
            'critical_paths': [],
            'recommended_actions': [],
            'tools_required': [],
            'estimated_time': '',
            'success_probability': ''
        }
        
        # Top 3 most dangerous paths
        for path in paths[:3]:
            path_plan = {
                'source': path.source,
                'target': path.target,
                'risk_score': path.risk_score,
                'steps': [],
                'tools': []
            }
            
            # Detail steps
            for i in range(len(path.path) - 1):
                step = {
                    'from': path.path[i],
                    'to': path.path[i + 1],
                    'relationship': path.relationship_types[i] if i < len(path.relationship_types) else 'Unknown',
                    'techniques': self._map_relationship_to_techniques(
                        path.relationship_types[i] if i < len(path.relationship_types) else 'Unknown'
                    )
                }
                path_plan['steps'].append(step)
            
            # Recommended tools
            path_plan['tools'] = self._recommend_tools_for_path(path)
            attack_plan['critical_paths'].append(path_plan)
        
        # General recommendations
        attack_plan['recommended_actions'] = self._generate_recommendations(paths)
        attack_plan['tools_required'] = list(set([tool for path in attack_plan['critical_paths'] for tool in path['tools']]))
        attack_plan['estimated_time'] = self._estimate_attack_time(paths)
        attack_plan['success_probability'] = self._estimate_success_probability(paths)
        
        return attack_plan
    
    def _map_relationship_to_techniques(self, relationship: str) -> List[str]:
        """Map relationships to attack techniques"""
        technique_map = {
            'AdminTo': ['Pass-the-Hash', 'Pass-the-Ticket', 'WMI Execution', 'PSExec'],
            'HasSession': ['Token Impersonation', 'Mimikatz', 'Session Hijacking'],
            'MemberOf': ['Group Policy Abuse', 'Privilege Escalation'],
            'ACL': ['ACL Abuse', 'DCSync Attack', 'Golden Ticket'],
            'GenericAll': ['Full Control Abuse', 'Password Reset', 'Add Member to Group'],
            'GenericWrite': ['Write Property Abuse', 'SPN Manipulation'],
            'WriteOwner': ['Ownership Takeover', 'ACL Modification'],
            'WriteDACL': ['ACL Modification', 'Privilege Escalation']
        }
        return technique_map.get(relationship, ['Unknown Technique'])
    
    def _recommend_tools_for_path(self, path: BloodhoundPath) -> List[str]:
        """Recommend tools for attack path"""
        tools = []
        
        for rel in path.relationship_types:
            if rel in ['AdminTo', 'HasSession']:
                tools.extend(['mimikatz', 'psexec', 'wmiexec', 'smbexec'])
            if rel in ['ACL', 'GenericAll', 'GenericWrite']:
                tools.extend(['bloodhound', 'powerview', 'ldapsearch'])
            if 'MemberOf' in rel:
                tools.extend(['net', 'powerview', 'admodule'])
        
        return list(set(tools))
    
    def _generate_recommendations(self, paths: List[BloodhoundPath]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if any(path.risk_score > 8.0 for path in paths):
            recommendations.append("CRITICAL: Immediate remediation required for high-risk attack paths")
        
        if any('AdminTo' in path.relationship_types for path in paths):
            recommendations.append("Review administrative privileges and implement Just-In-Time administration")
        
        if any('HasSession' in path.relationship_types for path in paths):
            recommendations.append("Implement session management controls and monitor for unusual logins")
        
        if any(path.path_length <= 3 for path in paths):
            recommendations.append("Short attack paths detected - review domain trust relationships")
        
        return recommendations
    
    def _estimate_attack_time(self, paths: List[BloodhoundPath]) -> str:
        """Estimate time to execute attack"""
        if not paths:
            return "Unknown"
        
        avg_length = sum(path.path_length for path in paths) / len(paths)
        
        if avg_length <= 3:
            return "1-2 hours (Short path)"
        elif avg_length <= 5:
            return "4-6 hours (Medium path)"
        else:
            return "8+ hours (Complex path)"
    
    def _estimate_success_probability(self, paths: List[BloodhoundPath]) -> str:
        """Estimate attack success probability"""
        if not paths:
            return "Unknown"
        
        avg_risk = sum(path.risk_score for path in paths) / len(paths)
        
        if avg_risk >= 8.0:
            return "High (90%+)"
        elif avg_risk >= 6.0:
            return "Medium (60-80%)"
        else:
            return "Low (30-50%)"

# ==================== TOOL INTEGRATIONS CONFIG ====================

TOOL_INTEGRATIONS = {
    # === NETWORK SCANNING ===
    "nmap_tcp_full": ToolIntegration(
        name="NMAP TCP Full Scan",
        framework=AttackFramework.NMAP,
        command="nmap",
        arguments=["-sS", "-sV", "-sC", "-O", "-p-", "-T4", "-v", "{target}"]
    ),
    
    "nmap_udp": ToolIntegration(
        name="NMAP UDP Scan", 
        framework=AttackFramework.NMAP,
        command="nmap",
        arguments=["-sU", "-sV", "--top-ports", "100", "-T4", "{target}"]
    ),
    
    "nmap_vuln_scan": ToolIntegration(
        name="NMAP Vulnerability Scan",
        framework=AttackFramework.NMAP,
        command="nmap",
        arguments=["-sS", "--script", "vuln", "-p", "{ports}", "{target}"]
    ),
    
    # === VULNERABILITY SCANNING ===
    "nuclei_scan": ToolIntegration(
        name="Nuclei Vulnerability Scan",
        framework=AttackFramework.NUCLEI,
        command="nuclei",
        arguments=["-u", "{url}", "-t", "cves/", "-t", "vulnerabilities/", "-t", "exposures/", "-rate-limit", "100"]
    ),
    
    "nikto_scan": ToolIntegration(
        name="Nikto Web Scan",
        framework=AttackFramework.NIKTO,
        command="nikto",
        arguments=["-h", "{url}", "-Tuning", "x", "-o", "nikto_{target}.html"]
    ),
    
    # === WEB APPLICATION ===
    "sqlmap_scan": ToolIntegration(
        name="SQLMap Automated Scan",
        framework=AttackFramework.SQLMAP,
        command="sqlmap",
        arguments=["-u", "{url}", "--batch", "--level=3", "--risk=2", "--crawl=10"]
    ),
    
    "gobuster_dir": ToolIntegration(
        name="Gobuster Directory Scan",
        framework=AttackFramework.GOBUSTER,
        command="gobuster",
        arguments=["dir", "-u", "{url}", "-w", "/usr/share/wordlists/dirb/common.txt", "-x", "php,html,txt"]
    ),
    
    "wpscan": ToolIntegration(
        name="WPScan WordPress Audit",
        framework=AttackFramework.WPSCAN,
        command="wpscan",
        arguments=["--url", "{url}", "--enumerate", "vp", "--plugins-detection", "aggressive"]
    ),
    
    # === ACTIVE DIRECTORY ===
    "crackmapexec_scan": ToolIntegration(
        name="CrackMapExec AD Scan",
        framework=AttackFramework.CRACKMAPEXEC,
        command="crackmapexec",
        arguments=["smb", "{target}", "--gen-relay-list", "targets.txt", "--shares", "--sessions", "--loggedon-users"]
    ),
    
    "bloodhound_collect": ToolIntegration(
        name="BloodHound Data Collection",
        framework=AttackFramework.BLOODHOUND,
        command="bloodhound-python",
        arguments=["-d", "{domain}", "-u", "{username}", "-p", "{password}", "-ns", "{dc_ip}", "--zip", "--dns-tcp"],
        requires_credentials=True
    ),
    
    "ldapsearch": ToolIntegration(
        name="LDAP Search",
        framework=AttackFramework.IMPACKET,
        command="ldapsearch",
        arguments=["-x", "-H", "ldap://{dc_ip}", "-b", "dc={domain},dc=com", "-s", "sub"]
    ),
    
    "enum4linux": ToolIntegration(
        name="Enum4Linux",
        framework=AttackFramework.IMPACKET,
        command="enum4linux",
        arguments=["-a", "-v", "{target}"]
    ),
    
    # === PASSWORD ATTACKS ===
    "hashcat_crack": ToolIntegration(
        name="Hashcat Password Cracking",
        framework=AttackFramework.HASHCAT,
        command="hashcat",
        arguments=["-m", "1000", "-a", "0", "--force", "hashes.txt", "/usr/share/wordlists/rockyou.txt"]
    ),
    
    "john_crack": ToolIntegration(
        name="John the Ripper",
        framework=AttackFramework.JOHN,
        command="john",
        arguments=["--wordlist=/usr/share/wordlists/rockyou.txt", "hashes.txt"]
    ),
    
    "hydra_ssh": ToolIntegration(
        name="Hydra SSH Bruteforce",
        framework=AttackFramework.HYDRA,
        command="hydra",
        arguments=["-L", "users.txt", "-P", "passwords.txt", "ssh://{target}"]
    ),
    
    # === EXPLOITATION ===
    "metasploit_scan": ToolIntegration(
        name="Metasploit AutoScan",
        framework=AttackFramework.METASPLOIT,
        command="msfconsole",
        arguments=["-q", "-r", "autoscan.rc"]
    ),
    
    # === EXPLOITDB ===
    "searchsploit": ToolIntegration(
        name="SearchSploit",
        framework=AttackFramework.EXPLOITDB,
        command="searchsploit",
        arguments=["--exclude=", "{query}"]
    ),

    # === IMPACKET TOOLS ===
    "getuserspns": ToolIntegration(
        name="GetUserSPNs (Kerberoasting)",
        framework=AttackFramework.IMPACKET,
        command="GetUserSPNs.py",
        arguments=["-dc-ip", "{dc_ip}", "-request", "{domain}/{username}:{password}"],
        requires_credentials=True
    ),
    
    "secretsdump": ToolIntegration(
        name="SecretsDump (DCSync)",
        framework=AttackFramework.IMPACKET,
        command="secretsdump.py",
        arguments=["-just-dc", "{domain}/{username}:{password}@{dc_ip}"],
        requires_credentials=True
    ),
    
    "psexec": ToolIntegration(
        name="PSEXEC",
        framework=AttackFramework.IMPACKET,
        command="psexec.py",
        arguments=["{domain}/{username}:{password}@{target}"],
        requires_credentials=True
    ),
    
    "smbclient": ToolIntegration(
        name="SMB Client",
        framework=AttackFramework.IMPACKET,
        command="smbclient.py",
        arguments=["-no-pass", "//{target}/IPC$"]
    )
}

# ==================== UTILITIES AND HELPERS ====================

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Logger:
    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger('autopentest')
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'{Color.BLUE}%(asctime)s{Color.END} {Color.GREEN}%(levelname)s{Color.END} {Color.WHITE}%(message)s{Color.END}'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(f"{Color.YELLOW}{message}{Color.END}")
    
    def error(self, message):
        self.logger.error(f"{Color.RED}{message}{Color.END}")
    
    def critical(self, message):
        self.logger.critical(f"{Color.RED}{Color.BOLD}{message}{Color.END}")
    
    def success(self, message):
        self.logger.info(f"{Color.GREEN}✓ {message}{Color.END}")

class NetworkUtils:
    @staticmethod
    def validate_ip(ip: str) -> bool:
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_cidr(cidr: str) -> bool:
        try:
            ipaddress.ip_network(cidr, strict=False)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def expand_targets(targets: List[str]) -> List[str]:
        expanded = []
        for target in targets:
            if NetworkUtils.validate_ip(target):
                expanded.append(target)
            elif NetworkUtils.validate_cidr(target):
                network = ipaddress.ip_network(target, strict=False)
                expanded.extend(str(ip) for ip in network.hosts())
            else:
                try:
                    ip = socket.gethostbyname(target)
                    expanded.append(ip)
                except socket.gaierror:
                    expanded.append(target)
        return expanded
    
    @staticmethod
    def is_port_open(ip: str, port: int, timeout: float = 1.0) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, port))
                return result == 0
        except:
            return False

class SecurityUtils:
    @staticmethod
    def generate_random_user_agent() -> str:
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]
        return random.choice(agents)
    
    @staticmethod
    def detect_web_technology(headers: Dict[str, str], body: str) -> List[str]:
        technologies = []
        
        # Header analysis
        server = headers.get('Server', '').lower()
        if 'apache' in server:
            technologies.append('Apache')
        if 'nginx' in server:
            technologies.append('Nginx')
        if 'iis' in server:
            technologies.append('IIS')
        
        # Body analysis
        body_lower = body.lower()
        tech_patterns = {
            'WordPress': ['wordpress', 'wp-content', 'wp-includes'],
            'Drupal': ['drupal', 'sites/all'],
            'Joomla': ['joomla', 'media/jui'],
            'React': ['react', 'react-dom'],
            'Vue.js': ['vue', 'vue.js'],
            'Angular': ['angular', 'ng-'],
            'jQuery': ['jquery'],
            'Bootstrap': ['bootstrap']
        }
        
        for tech, patterns in tech_patterns.items():
            if any(pattern in body_lower for pattern in patterns):
                technologies.append(tech)
        
        return technologies

# ==================== PORT SCANNERS ====================

class AsyncPortScanner:
    def __init__(self, max_concurrent: int = 500, timeout: float = 2.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = Logger()
    
    async def scan_port(self, ip: str, port: int) -> Optional[PortResult]:
        async with self.semaphore:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port),
                    timeout=self.timeout
                )
                
                # Port is open, grab banner
                banner = await self._grab_banner(reader, writer, ip, port)
                service = self._detect_service_by_port(port)
                
                writer.close()
                await writer.wait_closed()
                
                return PortResult(
                    port=port,
                    status=PortStatus.OPEN,
                    service=service,
                    banner=banner
                )
                
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                return None
            except Exception as e:
                self.logger.warning(f"Port scan error {port}: {e}")
                return None
    
    async def _grab_banner(self, reader, writer, ip: str, port: int) -> str:
        try:
            # Send basic requests for popular services
            if port == 22:  # SSH
                writer.write(b'SSH-2.0-Client\n')
                await writer.drain()
            elif port in [80, 443]:  # HTTP/HTTPS
                writer.write(b'HEAD / HTTP/1.0\r\n\r\n')
                await writer.drain()
            elif port == 21:  # FTP
                await reader.read(1024)  # Read greeting
            
            # Read response
            banner = await asyncio.wait_for(reader.read(1024), timeout=1.0)
            return banner.decode('utf-8', errors='ignore').strip()
        except:
            return ""
    
    def _detect_service_by_port(self, port: int) -> str:
        common_ports = {
            21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
            80: 'http', 110: 'pop3', 143: 'imap', 443: 'https', 993: 'imaps',
            995: 'pop3s', 1433: 'mssql', 1521: 'oracle', 3306: 'mysql',
            3389: 'rdp', 445: 'smb', 636: 'ldaps', 5985: 'winrm', 5986: 'winrm-ssl',
            5432: 'postgresql', 5900: 'vnc', 6379: 'redis'
        }
        return common_ports.get(port, 'unknown')
    
    async def scan_host(self, ip: str, ports: List[int]) -> List[PortResult]:
        self.logger.info(f"Scanning host {ip} ({len(ports)} ports)")
        
        tasks = [self.scan_port(ip, port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        open_ports = []
        for result in results:
            if isinstance(result, PortResult):
                open_ports.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Port scan task failed: {result}")
        
        self.logger.success(f"Found {len(open_ports)} open ports on {ip}")
        
        return open_ports

# ==================== SERVICE DETECTORS ====================

class HTTPDetector:
    def __init__(self):
        self.session = None
        self.logger = Logger()
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def detect_http_service(self, ip: str, port: int, ssl: bool = False) -> Dict[str, Any]:
        if not self.session:
            async with self:
                return await self.detect_http_service(ip, port, ssl)
        
        protocol = "https" if ssl else "http"
        url = f"{protocol}://{ip}:{port}"
        
        try:
            headers = {'User-Agent': SecurityUtils.generate_random_user_agent()}
            
            async with self.session.get(url, headers=headers, ssl=False) as response:
                # Basic information
                info = {
                    'url': url,
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'server': response.headers.get('Server', 'Unknown'),
                    'content_type': response.headers.get('Content-Type', 'Unknown'),
                    'technologies': [],
                    'title': '',
                    'forms': [],
                    'cookies': []
                }
                
                # Get body for analysis
                body = await response.text()
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', body, re.IGNORECASE)
                if title_match:
                    info['title'] = title_match.group(1).strip()
                
                # Detect technologies
                info['technologies'] = SecurityUtils.detect_web_technology(info['headers'], body)
                
                return info
                
        except Exception as e:
            self.logger.warning(f"HTTP service detection error {url}: {e}")
            return {'error': str(e)}

class WindowsServiceDetector:
    def __init__(self):
        self.logger = Logger()
    
    async def detect_windows_services(self, host: HostResult) -> Dict[str, Any]:
        """Detect Windows-specific services and features"""
        windows_info = {
            'is_windows': False,
            'windows_version': '',
            'netbios_name': '',
            'domain': '',
            'ad_services': []
        }
        
        # Check for Windows-specific ports
        windows_ports = [135, 139, 445, 3389, 5985, 5986]
        windows_ports_found = [port for port in host.ports if port.port in windows_ports]
        
        if len(windows_ports_found) >= 2:
            windows_info['is_windows'] = True
            
            # Analyze SMB for Windows version
            smb_port = next((port for port in host.ports if port.port == 445), None)
            if smb_port and smb_port.banner:
                windows_info['windows_version'] = self._parse_windows_version(smb_port.banner)
        
        return windows_info
    
    def _parse_windows_version(self, banner: str) -> str:
        """Parse Windows version from SMB banner"""
        if 'Windows 10' in banner or 'Windows 2016' in banner:
            return 'Windows 10/2016'
        elif 'Windows 8' in banner or 'Windows 2012' in banner:
            return 'Windows 8/2012'
        elif 'Windows 7' in banner or 'Windows 2008' in banner:
            return 'Windows 7/2008'
        elif 'Windows XP' in banner:
            return 'Windows XP/2003'
        else:
            return 'Unknown Windows'

# ==================== TOOL RUNNER ====================

class ToolRunner:
    def __init__(self):
        self.logger = Logger()
        self.tool_timeouts = {
            'nmap': 1800,
            'sqlmap': 3600,
            'hashcat': 86400,
            'bloodhound-python': 1800,
            'crackmapexec': 600,
            'metasploit': 3600,
            'searchsploit': 300
        }
    
    async def run_tool(self, tool_config: ToolIntegration, target: str, **kwargs) -> Dict[str, Any]:
        """Run tool with parameter substitution"""
        try:
            # Substitute values in arguments
            formatted_args = []
            for arg in tool_config.arguments:
                try:
                    # Replace placeholders with actual values
                    formatted_arg = arg
                    for key, value in kwargs.items():
                        placeholder = "{" + key + "}"
                        if placeholder in formatted_arg:
                            formatted_arg = formatted_arg.replace(placeholder, str(value))
                    
                    # Replace {target} placeholder
                    formatted_arg = formatted_arg.replace("{target}", target)
                    formatted_args.append(formatted_arg)
                except KeyError as e:
                    self.logger.warning(f"Missing parameter {e} for tool {tool_config.name}")
                    continue
            
            if not formatted_args:
                return {'success': False, 'error': 'No valid arguments after formatting'}
            
            # Tool-specific timeout
            timeout = self.tool_timeouts.get(tool_config.command, 300)
            
            # Run process
            cmd = [tool_config.command] + formatted_args
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                return {
                    'success': process.returncode == 0,
                    'stdout': stdout.decode('utf-8', errors='ignore') if stdout else '',
                    'stderr': stderr.decode('utf-8', errors='ignore') if stderr else '',
                    'returncode': process.returncode,
                    'command': ' '.join(cmd)
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'success': False,
                    'error': f"Tool timed out after {timeout} seconds",
                    'command': ' '.join(cmd)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': tool_config.command
            }
    
    def generate_tool_commands(self, scan_result: ScanResult, framework: AttackFramework = None) -> List[ToolIntegration]:
        """Generate tool commands based on scan results"""
        commands = []
        
        for host in scan_result.hosts:
            open_ports = [port.port for port in host.ports if port.status == PortStatus.OPEN]
            
            # NMAP commands
            if open_ports:
                commands.append(TOOL_INTEGRATIONS["nmap_tcp_full"])
                commands.append(TOOL_INTEGRATIONS["nmap_vuln_scan"])
            
            # Web scanners
            web_ports = [port for port in host.ports if port.service in ['http', 'https']]
            if web_ports:
                commands.append(TOOL_INTEGRATIONS["nuclei_scan"])
                commands.append(TOOL_INTEGRATIONS["nikto_scan"])
                commands.append(TOOL_INTEGRATIONS["gobuster_dir"])
                
                # Check for WordPress
                for port in web_ports:
                    service_key = f"{port.service}_{port.port}"
                    if service_key in host.services:
                        service_info = host.services[service_key]
                        if isinstance(service_info, dict) and 'technologies' in service_info:
                            technologies = service_info.get('technologies', [])
                            if any('wordpress' in tech.lower() for tech in technologies):
                                commands.append(TOOL_INTEGRATIONS["wpscan"])
            
            # AD tools
            ad_ports = [port for port in host.ports if port.port in [53, 88, 389, 445, 636]]
            if ad_ports:
                commands.append(TOOL_INTEGRATIONS["crackmapexec_scan"])
                commands.append(TOOL_INTEGRATIONS["enum4linux"])
                commands.append(TOOL_INTEGRATIONS["smbclient"])
            
            # Password attacks
            if any(port.service in ['ssh', 'ftp', 'telnet'] for port in host.ports):
                commands.append(TOOL_INTEGRATIONS["hydra_ssh"])
            
            # ExploitDB search
            if any(port.status == PortStatus.OPEN for port in host.ports):
                commands.append(TOOL_INTEGRATIONS["searchsploit"])
        
        if framework:
            commands = [cmd for cmd in commands if cmd.framework == framework]
        
        # Remove duplicates using a set (now that ToolIntegration is hashable)
        unique_commands = []
        seen = set()
        for cmd in commands:
            if cmd not in seen:
                seen.add(cmd)
                unique_commands.append(cmd)
        
        return unique_commands

class IntegrationManager:
    def __init__(self):
        self.tool_runner = ToolRunner()
        self.ad_scanner = ADScanner(self.tool_runner)
        self.bloodhound_parser = BloodhoundParser()
        self.exploitdb = ExploitDBIntegration()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.logger = Logger()
    
    async def run_all_integrations(self, scan_result: ScanResult, credentials: Dict[str, str] = None) -> Dict[str, Any]:
        """Run all tool integrations based on scan results"""
        integrations = {
            'nmap': [],
            'vulnerability_scanners': [],
            'web_tools': [],
            'ad_tools': [],
            'exploitation': [],
            'password_attacks': [],
            'exploitdb': []
        }
        
        # Generate commands for each framework
        all_tools = self.tool_runner.generate_tool_commands(scan_result)
        
        for tool in all_tools:
            category = self._categorize_tool(tool)
            
            # Run for each relevant host
            for host in scan_result.hosts:
                if self._should_run_tool(tool, host):
                    # Prepare credentials if needed
                    tool_kwargs = {}
                    if tool.requires_credentials and credentials:
                        tool_kwargs.update(credentials)
                    
                    # Special handling for searchsploit
                    if tool.framework == AttackFramework.EXPLOITDB:
                        # Search for exploits based on host services
                        for port in host.ports:
                            if port.status == PortStatus.OPEN:
                                query = f"{port.service}"
                                if port.version:
                                    query += f" {port.version}"
                                tool_kwargs['query'] = query
                                result = await self.tool_runner.run_tool(tool, host.ip, **tool_kwargs)
                                integrations[category].append({
                                    'tool': f"{tool.name} - {query}",
                                    'target': host.ip,
                                    'result': result
                                })
                    else:
                        # Handle web tools
                        if tool.framework in [AttackFramework.NUCLEI, AttackFramework.NIKTO, AttackFramework.GOBUSTER]:
                            web_ports = [port for port in host.ports if port.service in ['http', 'https']]
                            for port in web_ports:
                                protocol = 'https' if port.service == 'https' else 'http'
                                url = f"{protocol}://{host.ip}:{port.port}"
                                tool_kwargs['url'] = url
                                result = await self.tool_runner.run_tool(tool, host.ip, **tool_kwargs)
                                integrations[category].append({
                                    'tool': f"{tool.name} - {url}",
                                    'target': host.ip,
                                    'result': result
                                })
                        else:
                            result = await self.tool_runner.run_tool(tool, host.ip, **tool_kwargs)
                            integrations[category].append({
                                'tool': tool.name,
                                'target': host.ip,
                                'result': result
                            })
        
        return integrations
    
    def _categorize_tool(self, tool: ToolIntegration) -> str:
        """Categorize tools"""
        categorization = {
            AttackFramework.NMAP: 'nmap',
            AttackFramework.NUCLEI: 'vulnerability_scanners',
            AttackFramework.NIKTO: 'vulnerability_scanners',
            AttackFramework.SQLMAP: 'web_tools',
            AttackFramework.GOBUSTER: 'web_tools',
            AttackFramework.WPSCAN: 'web_tools',
            AttackFramework.CRACKMAPEXEC: 'ad_tools',
            AttackFramework.BLOODHOUND: 'ad_tools',
            AttackFramework.IMPACKET: 'ad_tools',
            AttackFramework.HASHCAT: 'password_attacks',
            AttackFramework.JOHN: 'password_attacks',
            AttackFramework.HYDRA: 'password_attacks',
            AttackFramework.METASPLOIT: 'exploitation',
            AttackFramework.EXPLOITDB: 'exploitdb'
        }
        return categorization.get(tool.framework, 'other')
    
    def _should_run_tool(self, tool: ToolIntegration, host: HostResult) -> bool:
        """Determine if tool should run for host"""
        open_ports = [port.port for port in host.ports if port.status == PortStatus.OPEN]
        
        # NMAP - for all hosts with open ports
        if tool.framework == AttackFramework.NMAP:
            return bool(open_ports)
        
        # Web tools - for HTTP/HTTPS
        if tool.framework in [AttackFramework.NUCLEI, AttackFramework.NIKTO, AttackFramework.SQLMAP, AttackFramework.GOBUSTER]:
            return any(port.service in ['http', 'https'] for port in host.ports)
        
        # WordPress tools
        if tool.framework == AttackFramework.WPSCAN:
            for port in host.ports:
                if port.service in ['http', 'https']:
                    service_key = f"{port.service}_{port.port}"
                    if service_key in host.services:
                        service_info = host.services[service_key]
                        if isinstance(service_info, dict) and 'technologies' in service_info:
                            technologies = service_info.get('technologies', [])
                            if any('wordpress' in tech.lower() for tech in technologies):
                                return True
            return False
        
        # AD tools
        if tool.framework in [AttackFramework.CRACKMAPEXEC, AttackFramework.BLOODHOUND, AttackFramework.IMPACKET]:
            return any(port.port in [53, 88, 389, 445, 636] for port in host.ports)
        
        # Password attacks
        if tool.framework in [AttackFramework.HYDRA]:
            return any(port.service in ['ssh', 'ftp', 'telnet'] for port in host.ports)
        
        # ExploitDB - for all hosts with open ports
        if tool.framework == AttackFramework.EXPLOITDB:
            return bool(open_ports)
        
        return True

# ==================== ACTIVE DIRECTORY MODULES ====================

class ADScanner:
    def __init__(self, tool_runner: ToolRunner):
        self.tool_runner = tool_runner
        self.logger = Logger()
    
    async def detect_domain_controller(self, target: str) -> Optional[ADHostResult]:
        """Detect Domain Controller"""
        try:
            # Check AD ports
            ad_ports = [53, 88, 389, 445, 464, 636, 3268, 3269]
            open_ad_ports = []
            
            for port in ad_ports:
                if NetworkUtils.is_port_open(target, port, timeout=2.0):
                    open_ad_ports.append(port)
            
            # If we have multiple AD ports, likely a DC
            if len(open_ad_ports) >= 3:
                ad_host = ADHostResult(
                    ip=target, 
                    is_domain_controller=True,
                    ad_roles=['domain_controller']
                )
                
                self.logger.success(f"Potential Domain Controller detected: {target}")
                return ad_host
                
        except Exception as e:
            self.logger.error(f"AD detection error for {target}: {e}")
        
        return None
    
    async def perform_ad_recon(self, domain: str, dc_ip: str, username: str = "", password: str = "") -> Dict[str, Any]:
        """Complete Active Directory reconnaissance"""
        results = {}
        
        # Basic AD enumeration without credentials
        if not username or not password:
            self.logger.info("Performing AD reconnaissance without credentials")
            
            # Enum4Linux
            enum_result = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["enum4linux"],
                dc_ip
            )
            results['enum4linux'] = enum_result
            
            # CrackMapExec basic scan
            cme_result = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["crackmapexec_scan"],
                dc_ip
            )
            results['crackmapexec'] = cme_result
        
        else:
            self.logger.info("Performing credentialed AD reconnaissance")
            
            # Bloodhound data collection
            bloodhound_result = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["bloodhound_collect"],
                dc_ip,
                domain=domain,
                username=username,
                password=password,
                dc_ip=dc_ip
            )
            results['bloodhound'] = bloodhound_result
            
            # Kerberoasting attack
            kerberoast_result = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["getuserspns"],
                dc_ip,
                domain=domain,
                username=username,
                password=password,
                dc_ip=dc_ip
            )
            results['kerberoasting'] = kerberoast_result
            
            # DCSync attempt
            dcsync_result = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["secretsdump"],
                dc_ip,
                domain=domain,
                username=username,
                password=password,
                dc_ip=dc_ip
            )
            results['secretsdump'] = dcsync_result
        
        return results
    
    async def check_ad_vulnerabilities(self, target: str) -> List[Vulnerability]:
        """Check AD-specific vulnerabilities"""
        vulnerabilities = []
        
        # Check for common AD vulnerabilities
        vuln_checks = [
            self._check_zerologon,
            self._check_eternalblue,
            self._check_smbghost
        ]
        
        for check in vuln_checks:
            try:
                vuln = await check(target)
                if vuln:
                    vulnerabilities.append(vuln)
            except Exception as e:
                self.logger.warning(f"Vulnerability check failed: {e}")
        
        return vulnerabilities
    
    async def _check_zerologon(self, target: str) -> Optional[Vulnerability]:
        """Check for Zerologon vulnerability"""
        try:
            # Use nmap script to check
            result = await self.tool_runner.run_tool(
                ToolIntegration(
                    name="Zerologon Check",
                    framework=AttackFramework.NMAP,
                    command="nmap",
                    arguments=["-p", "445", "--script", "smb-vuln-cve-2020-1472", "{target}"]
                ),
                target
            )
            
            if 'VULNERABLE' in result.get('stdout', ''):
                return Vulnerability(
                    id="zerologon",
                    name="Zerologon (CVE-2020-1472)",
                    description="Critical Netlogon privilege escalation vulnerability",
                    risk_level=RiskLevel.CRITICAL,
                    cvss_score=10.0,
                    exploit_available=True,
                    remediation="Apply Microsoft patch KB4557222",
                    service="smb",
                    port=445
                )
        except Exception as e:
            self.logger.warning(f"Zerologon check failed: {e}")
        
        return None
    
    async def _check_eternalblue(self, target: str) -> Optional[Vulnerability]:
        """Check for EternalBlue vulnerability"""
        try:
            result = await self.tool_runner.run_tool(
                ToolIntegration(
                    name="EternalBlue Check",
                    framework=AttackFramework.NMAP,
                    command="nmap",
                    arguments=["-p", "445", "--script", "smb-vuln-ms17-010", "{target}"]
                ),
                target
            )
            
            if 'VULNERABLE' in result.get('stdout', ''):
                return Vulnerability(
                    id="eternalblue",
                    name="EternalBlue (MS17-010)",
                    description="SMBv1 remote code execution vulnerability",
                    risk_level=RiskLevel.CRITICAL,
                    cvss_score=8.5,
                    exploit_available=True,
                    service="smb",
                    port=445
                )
        except Exception as e:
            self.logger.warning(f"EternalBlue check failed: {e}")
        
        return None
    
    async def _check_smbghost(self, target: str) -> Optional[Vulnerability]:
        """Check for SMBGhost vulnerability"""
        try:
            result = await self.tool_runner.run_tool(
                ToolIntegration(
                    name="SMBGhost Check",
                    framework=AttackFramework.NMAP,
                    command="nmap",
                    arguments=["-p", "445", "--script", "smb-vuln-cve-2020-0796", "{target}"]
                ),
                target
            )
            
            if 'VULNERABLE' in result.get('stdout', ''):
                return Vulnerability(
                    id="smbghost",
                    name="SMBGhost (CVE-2020-0796)",
                    description="SMBv3 compression buffer overflow",
                    risk_level=RiskLevel.HIGH,
                    cvss_score=8.8,
                    exploit_available=True,
                    service="smb",
                    port=445
                )
        except Exception as e:
            self.logger.warning(f"SMBGhost check failed: {e}")
        
        return None

# ==================== SCANNER MANAGER ====================

class ScannerManager:
    def __init__(self):
        self.port_scanner = AsyncPortScanner()
        self.http_detector = HTTPDetector()
        self.windows_detector = WindowsServiceDetector()
        self.ad_scanner = ADScanner(ToolRunner())
        self.exploitdb = ExploitDBIntegration()
        self.logger = Logger()
    
    async def comprehensive_scan(self, ip: str, ports: List[int] = None) -> HostResult:
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 5985, 5986, 6379]
        
        host_result = HostResult(ip=ip)
        
        try:
            # Port scanning
            self.logger.info(f"Starting comprehensive scan for host {ip}")
            open_ports = await self.port_scanner.scan_host(ip, ports)
            host_result.ports = open_ports
            
            # Service detection for open ports
            service_tasks = []
            for port_result in open_ports:
                if port_result.service in ['http', 'https']:
                    task = self._detect_http_service(ip, port_result)
                else:
                    task = self._analyze_generic_service(ip, port_result)
                service_tasks.append(task)
            
            service_results = await asyncio.gather(*service_tasks, return_exceptions=True)
            
            # Update service information
            for port_result, service_info in zip(open_ports, service_results):
                if service_info and not isinstance(service_info, Exception):
                    host_result.services[f"{port_result.service}_{port_result.port}"] = service_info
            
            # Windows detection
            windows_info = await self.windows_detector.detect_windows_services(host_result)
            host_result.is_windows = windows_info['is_windows']
            
            # AD detection
            if any(port.port in [53, 88, 389, 445, 636] for port in open_ports):
                ad_host = await self.ad_scanner.detect_domain_controller(ip)
                if ad_host:
                    host_result.is_domain_controller = True
            
            # Vulnerability scanning
            ad_vulnerabilities = await self.ad_scanner.check_ad_vulnerabilities(ip)
            host_result.vulnerabilities.extend(ad_vulnerabilities)
            
            # ExploitDB search for found services
            exploits = self.exploitdb.generate_exploit_suggestions(host_result)
            host_result.exploits = exploits
            
            # Risk calculation
            host_result.risk_score = self._calculate_host_risk(host_result)
            
            self.logger.success(f"Completed scan for host {ip}. Risk score: {host_result.risk_score:.2f}")
            if exploits:
                self.logger.info(f"Found {len(exploits)} potential exploits")
            
        except Exception as e:
            self.logger.error(f"Error scanning host {ip}: {e}")
        
        return host_result
    
    async def _detect_http_service(self, ip: str, port_result: PortResult) -> Dict[str, Any]:
        try:
            ssl = port_result.service == 'https'
            async with self.http_detector as detector:
                return await detector.detect_http_service(ip, port_result.port, ssl)
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_generic_service(self, ip: str, port_result: PortResult) -> Dict[str, Any]:
        return {
            'service': port_result.service,
            'port': port_result.port,
            'banner': port_result.banner,
            'detected': True
        }
    
    def _calculate_host_risk(self, host: HostResult) -> float:
        risk_score = 0.0
        
        # Base risk for open ports
        risk_score += len(host.ports) * 0.1
        
        # Increased risk for specific services
        high_risk_services = ['ssh', 'ftp', 'telnet', 'smb', 'rdp']
        medium_risk_services = ['http', 'https', 'mysql', 'postgresql']
        
        for port in host.ports:
            if port.service in high_risk_services:
                risk_score += 0.5
            elif port.service in medium_risk_services:
                risk_score += 0.3
        
        # Additional risk for Windows/AD
        if host.is_windows:
            risk_score += 1.0
        if host.is_domain_controller:
            risk_score += 2.0
        
        # Additional risk for vulnerabilities
        for vuln in host.vulnerabilities:
            if hasattr(vuln, 'risk_level'):
                risk_score += vuln.risk_level.value
            elif isinstance(vuln, dict) and 'severity' in vuln:
                severity_weights = {'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}
                risk_score += severity_weights.get(vuln.get('severity', 'low'), 0.5)
        
        # Additional risk for exploits
        for exploit in host.exploits:
            risk_score += exploit.risk_level.value * 0.3
        
        return min(risk_score, 10.0)

# ==================== VULNERABILITY SCANNER ====================

class VulnerabilityScanner:
    def __init__(self):
        self.tool_runner = ToolRunner()
        self.logger = Logger()
    
    async def scan_host_vulnerabilities(self, host: HostResult) -> List[Vulnerability]:
        """Scan host for vulnerabilities using multiple tools"""
        vulnerabilities = []
        
        # NMAP vulnerability scripts
        if any(port.status == PortStatus.OPEN for port in host.ports):
            open_ports_str = ",".join(str(port.port) for port in host.ports if port.status == PortStatus.OPEN)
            vuln_scan = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["nmap_vuln_scan"],
                host.ip,
                ports=open_ports_str
            )
            
            if vuln_scan['success']:
                nmap_vulns = self._parse_nmap_vulnerabilities(vuln_scan['stdout'], host)
                vulnerabilities.extend(nmap_vulns)
        
        # Nuclei scan for web services
        web_ports = [port for port in host.ports if port.service in ['http', 'https']]
        for port in web_ports:
            protocol = 'https' if port.service == 'https' else 'http'
            url = f"{protocol}://{host.ip}:{port.port}"
            
            nuclei_scan = await self.tool_runner.run_tool(
                TOOL_INTEGRATIONS["nuclei_scan"],
                host.ip,
                url=url
            )
            
            if nuclei_scan['success']:
                nuclei_vulns = self._parse_nuclei_vulnerabilities(nuclei_scan['stdout'], host, port)
                vulnerabilities.extend(nuclei_vulns)
        
        return vulnerabilities
    
    def _parse_nmap_vulnerabilities(self, nmap_output: str, host: HostResult) -> List[Vulnerability]:
        """Parse vulnerabilities from NMAP output"""
        vulnerabilities = []
        
        # Parse common vulnerability patterns
        vuln_patterns = {
            r'VULNERABLE:\s*(.+)': RiskLevel.HIGH,
            r'CVE-\d{4}-\d+': RiskLevel.MEDIUM,
            r'MS\d{2}-\d+': RiskLevel.MEDIUM,
        }
        
        for line in nmap_output.split('\n'):
            for pattern, risk_level in vuln_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = Vulnerability(
                        id=str(uuid.uuid4())[:8],
                        name=f"NMAP Finding: {line.strip()}",
                        description=f"Vulnerability detected by NMAP: {line.strip()}",
                        risk_level=risk_level,
                        service="multiple",
                        port=0
                    )
                    vulnerabilities.append(vulnerability)
                    break
        
        return vulnerabilities
    
    def _parse_nuclei_vulnerabilities(self, nuclei_output: str, host: HostResult, port: PortResult) -> List[Vulnerability]:
        """Parse vulnerabilities from Nuclei output"""
        vulnerabilities = []
        
        # Parse Nuclei JSON lines output
        for line in nuclei_output.split('\n'):
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    finding = json.loads(line)
                    
                    # Extract CVE if present
                    cve_id = ""
                    if 'info' in finding and 'classification' in finding['info']:
                        cve_data = finding['info']['classification'].get('cve-id', [])
                        if cve_data and len(cve_data) > 0:
                            cve_id = cve_data[0]
                    
                    severity = finding.get('info', {}).get('severity', 'unknown')
                    risk_level = self._map_severity_to_risk(severity)
                    
                    vulnerability = Vulnerability(
                        id=finding.get('template-id', str(uuid.uuid4())[:8]),
                        name=finding.get('info', {}).get('name', 'Unknown Vulnerability'),
                        description=finding.get('info', {}).get('description', 'No description available'),
                        risk_level=risk_level,
                        cvss_score=float(finding.get('info', {}).get('classification', {}).get('cvss-score', 0.0)),
                        cve_id=cve_id,
                        service=port.service,
                        port=port.port,
                        nuclei_template=finding.get('template-id', '')
                    )
                    vulnerabilities.append(vulnerability)
                    
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue
        
        return vulnerabilities
    
    def _map_severity_to_risk(self, severity: str) -> RiskLevel:
        """Map Nuclei severity to RiskLevel"""
        severity_map = {
            'critical': RiskLevel.CRITICAL,
            'high': RiskLevel.HIGH,
            'medium': RiskLevel.MEDIUM,
            'low': RiskLevel.LOW,
            'info': RiskLevel.INFO
        }
        return severity_map.get(severity.lower(), RiskLevel.INFO)

# ==================== INTEGRATION MANAGER ====================

class IntegrationManager:
    def __init__(self):
        self.tool_runner = ToolRunner()
        self.ad_scanner = ADScanner(self.tool_runner)
        self.bloodhound_parser = BloodhoundParser()
        self.exploitdb = ExploitDBIntegration()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.logger = Logger()
    
    async def run_all_integrations(self, scan_result: ScanResult, credentials: Dict[str, str] = None) -> Dict[str, Any]:
        """Run all tool integrations based on scan results"""
        integrations = {
            'nmap': [],
            'vulnerability_scanners': [],
            'web_tools': [],
            'ad_tools': [],
            'exploitation': [],
            'password_attacks': [],
            'exploitdb': []
        }
        
        # Generate commands for each framework
        all_tools = self.tool_runner.generate_tool_commands(scan_result)
        
        for tool in all_tools:
            category = self._categorize_tool(tool)
            
            # Run for each relevant host
            for host in scan_result.hosts:
                if self._should_run_tool(tool, host):
                    # Prepare credentials if needed
                    tool_kwargs = {}
                    if tool.requires_credentials and credentials:
                        tool_kwargs.update(credentials)
                    
                    # Special handling for searchsploit
                    if tool.framework == AttackFramework.EXPLOITDB:
                        # Search for exploits based on host services
                        for port in host.ports:
                            if port.status == PortStatus.OPEN:
                                query = f"{port.service}"
                                if port.version:
                                    query += f" {port.version}"
                                tool_kwargs['query'] = query
                                result = await self.tool_runner.run_tool(tool, host.ip, **tool_kwargs)
                                integrations[category].append({
                                    'tool': f"{tool.name} - {query}",
                                    'target': host.ip,
                                    'result': result
                                })
                    else:
                        result = await self.tool_runner.run_tool(tool, host.ip, **tool_kwargs)
                        integrations[category].append({
                            'tool': tool.name,
                            'target': host.ip,
                            'result': result
                        })
        
        return integrations
    
    def _categorize_tool(self, tool: ToolIntegration) -> str:
        """Categorize tools"""
        categorization = {
            AttackFramework.NMAP: 'nmap',
            AttackFramework.NUCLEI: 'vulnerability_scanners',
            AttackFramework.NIKTO: 'vulnerability_scanners',
            AttackFramework.SQLMAP: 'web_tools',
            AttackFramework.GOBUSTER: 'web_tools',
            AttackFramework.WPSCAN: 'web_tools',
            AttackFramework.CRACKMAPEXEC: 'ad_tools',
            AttackFramework.BLOODHOUND: 'ad_tools',
            AttackFramework.IMPACKET: 'ad_tools',
            AttackFramework.HASHCAT: 'password_attacks',
            AttackFramework.JOHN: 'password_attacks',
            AttackFramework.HYDRA: 'password_attacks',
            AttackFramework.METASPLOIT: 'exploitation',
            AttackFramework.EXPLOITDB: 'exploitdb'
        }
        return categorization.get(tool.framework, 'other')
    
    def _should_run_tool(self, tool: ToolIntegration, host: HostResult) -> bool:
        """Determine if tool should run for host"""
        open_ports = [port.port for port in host.ports if port.status == PortStatus.OPEN]
        
        # NMAP - for all hosts with open ports
        if tool.framework == AttackFramework.NMAP:
            return bool(open_ports)
        
        # Web tools - for HTTP/HTTPS
        if tool.framework in [AttackFramework.NUCLEI, AttackFramework.NIKTO, AttackFramework.SQLMAP, AttackFramework.GOBUSTER]:
            return any(port.service in ['http', 'https'] for port in host.ports)
        
        # WordPress tools
        if tool.framework == AttackFramework.WPSCAN:
            for port in host.ports:
                if port.service in ['http', 'https']:
                    service_key = f"{port.service}_{port.port}"
                    if service_key in host.services:
                        service_info = host.services[service_key]
                        if isinstance(service_info, dict) and 'technologies' in service_info:
                            technologies = service_info.get('technologies', [])
                            if any('wordpress' in tech.lower() for tech in technologies):
                                return True
            return False
        
        # AD tools
        if tool.framework in [AttackFramework.CRACKMAPEXEC, AttackFramework.BLOODHOUND, AttackFramework.IMPACKET]:
            return any(port.port in [53, 88, 389, 445, 636] for port in host.ports)
        
        # Password attacks
        if tool.framework in [AttackFramework.HYDRA]:
            return any(port.service in ['ssh', 'ftp', 'telnet'] for port in host.ports)
        
        # ExploitDB - for all hosts with open ports
        if tool.framework == AttackFramework.EXPLOITDB:
            return bool(open_ports)
        
        return True
    
    async def analyze_bloodhound_data(self, zip_path: str, target_domain: str = "Domain Admins") -> Dict[str, Any]:
        """Analyze Bloodhound data and find attack paths"""
        if not SIMDJSON_AVAILABLE:
            self.logger.warning("simdjson not available, Bloodhound analysis disabled")
            return {}
        
        try:
            # Parse Bloodhound data
            bloodhound_data = self.bloodhound_parser.parse_bloodhound_data(zip_path)
            
            # Find attack paths
            attack_paths = self.bloodhound_parser.find_attack_paths(target_node=target_domain)
            
            # Generate attack plan
            attack_plan = self.bloodhound_parser.generate_attack_plan(attack_paths)
            
            self.logger.success(f"Bloodhound analysis complete: {len(attack_paths)} attack paths found")
            
            return {
                'bloodhound_data': bloodhound_data,
                'attack_paths': [asdict(path) for path in attack_paths],
                'attack_plan': attack_plan
            }
            
        except Exception as e:
            self.logger.error(f"Bloodhound analysis error: {e}")
            return {}
    
    def search_exploits_for_host(self, host: HostResult) -> List[Exploit]:
        """Search exploits for specific host"""
        return self.exploitdb.generate_exploit_suggestions(host)
    
    def download_exploit(self, exploit_id: str, output_dir: str = "exploits") -> Optional[str]:
        """Download exploit by ID"""
        return self.exploitdb.download_exploit(exploit_id, output_dir)

# ==================== MAIN AUTO PENTEST ASSISTANT ====================

class AutoPentestAssistant:
    def __init__(self):
        self.scanner_manager = ScannerManager()
        self.integration_manager = IntegrationManager()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.tool_runner = ToolRunner()
        self.logger = Logger()
    
    async def comprehensive_scan_with_integrations(self, targets: List[str], ports: List[int] = None, 
                                                 credentials: Dict[str, str] = None) -> Dict[str, Any]:
        """Complete scanning with all integrations"""
        start_time = datetime.now()
        
        self.logger.info(f"Starting comprehensive scan for {len(targets)} targets")
        
        # Expand targets (CIDR, domains, etc.)
        expanded_targets = NetworkUtils.expand_targets(targets)
        self.logger.info(f"Expanded to {len(expanded_targets)} IP addresses")
        
        # Scan each host
        scan_tasks = []
        for target in expanded_targets:
            if NetworkUtils.validate_ip(target):
                task = self.scanner_manager.comprehensive_scan(target, ports)
                scan_tasks.append(task)
        
        hosts = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Filter hosts with results and handle exceptions
        valid_hosts = []
        for host in hosts:
            if isinstance(host, HostResult) and host.ports:
                valid_hosts.append(host)
            elif isinstance(host, Exception):
                self.logger.warning(f"Host scan failed: {host}")
        
        # AD detection
        ad_hosts = []
        for host in valid_hosts:
            if host.is_domain_controller or any(port.port in [53, 88, 389, 445, 636] for port in host.ports):
                ad_host = await self.integration_manager.ad_scanner.detect_domain_controller(host.ip)
                if ad_host:
                    ad_hosts.append(ad_host)
        
        # Vulnerability scanning
        for host in valid_hosts:
            vulnerabilities = await self.vulnerability_scanner.scan_host_vulnerabilities(host)
            host.vulnerabilities.extend(vulnerabilities)
            # Recalculate risk with vulnerabilities
            host.risk_score = self._calculate_enhanced_risk(host)
        
        end_time = datetime.now()
        
        scan_result = ScanResult(
            targets=targets,
            hosts=valid_hosts,
            ad_hosts=ad_hosts,
            start_time=start_time,
            end_time=end_time,
            scan_config={
                'ports': ports,
                'max_concurrent': 500,
                'timeout': 2.0
            },
            statistics={
                'total_hosts_scanned': len(expanded_targets),
                'hosts_with_open_ports': len(valid_hosts),
                'ad_hosts_found': len(ad_hosts),
                'total_open_ports': sum(len(host.ports) for host in valid_hosts),
                'total_vulnerabilities': sum(len(host.vulnerabilities) for host in valid_hosts),
                'total_exploits': sum(len(host.exploits) for host in valid_hosts),
                'scan_duration': str(end_time - start_time)
            }
        )
        
        # Run tool integrations
        self.logger.info("Running tool integrations...")
        tool_results = await self.integration_manager.run_all_integrations(scan_result, credentials)
        scan_result.tool_results = tool_results
        
        # Collect all exploits
        all_exploits = []
        for host in valid_hosts:
            all_exploits.extend(host.exploits)
        scan_result.exploits_found = all_exploits
        
        return {
            'scan_result': scan_result,
            'tool_results': tool_results
        }
    
    async def analyze_bloodhound_data(self, zip_path: str, target_domain: str = "Domain Admins") -> Dict[str, Any]:
        """Analyze Bloodhound data and return attack paths"""
        return await self.integration_manager.analyze_bloodhound_data(zip_path, target_domain)
    
    def search_exploits(self, query: str, service: str = "", port: int = 0) -> List[Exploit]:
        """Search exploits in ExploitDB"""
        return self.integration_manager.exploitdb.search_exploits(query, service, port)
    
    def download_exploit(self, exploit_id: str, output_dir: str = "exploits") -> Optional[str]:
        """Download exploit by ID"""
        return self.integration_manager.download_exploit(exploit_id, output_dir)
    
    def _calculate_enhanced_risk(self, host: HostResult) -> float:
        """Calculate enhanced risk score considering vulnerabilities"""
        base_risk = 0.0
        
        # Base risk for open ports
        base_risk += len(host.ports) * 0.1
        
        # Service-specific risks
        high_risk_services = ['ssh', 'ftp', 'telnet', 'smb', 'rdp']
        medium_risk_services = ['http', 'https', 'mysql', 'postgresql']
        
        for port in host.ports:
            if port.service in high_risk_services:
                base_risk += 0.5
            elif port.service in medium_risk_services:
                base_risk += 0.3
        
        # Windows/AD risks
        if host.is_windows:
            base_risk += 1.0
        if host.is_domain_controller:
            base_risk += 2.0
        
        # Vulnerability risks
        for vuln in host.vulnerabilities:
            if hasattr(vuln, 'risk_level'):
                base_risk += vuln.risk_level.value * 0.5
        
        # Exploit risks
        for exploit in host.exploits:
            base_risk += exploit.risk_level.value * 0.3
        
        return min(base_risk, 10.0)

# ==================== MAIN EXECUTION ====================

async def main():
    parser = argparse.ArgumentParser(description='🚀 AutoPentest Assistant ULTIMATE v7.1')
    parser.add_argument('targets', nargs='+', help='Targets to scan (IP, CIDR, domain)')
    parser.add_argument('-p', '--ports', help='Ports to scan (default: common ports)')
    parser.add_argument('--ad-scan', action='store_true', help='Enable AD scanning')
    parser.add_argument('--run-tools', action='store_true', help='Run all tool integrations')
    parser.add_argument('--bloodhound', help='Analyze Bloodhound ZIP file')
    parser.add_argument('--bloodhound-target', default='Domain Admins', help='Bloodhound target group (default: Domain Admins)')
    parser.add_argument('--search-exploits', help='Search ExploitDB for specific query')
    parser.add_argument('--download-exploit', help='Download exploit by ID')
    parser.add_argument('--generate-reports', action='store_true', help='Generate comprehensive reports')
    parser.add_argument('--username', help='Username for credentialed scans')
    parser.add_argument('--password', help='Password for credentialed scans')
    parser.add_argument('--domain', help='Domain for AD scans')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    logger = Logger()
    logger.info("🚀 Starting AutoPentest Assistant ULTIMATE v7.1")
    
    # Check dependencies
    if not SIMDJSON_AVAILABLE:
        logger.warning("simdjson not available - Bloodhound parsing will be slower")
    if not NETWORKX_AVAILABLE:
        logger.warning("networkx not available - Bloodhound path analysis disabled")
    
    try:
        # Parse ports
        ports = None
        if args.ports:
            if '-' in args.ports:
                start, end = map(int, args.ports.split('-'))
                ports = list(range(start, end + 1))
            else:
                ports = [int(p) for p in args.ports.split(',')]
        
        # Prepare credentials
        credentials = {}
        if args.username and args.password:
            credentials = {
                'username': args.username,
                'password': args.password,
                'domain': args.domain or 'domain'
            }
        
        assistant = AutoPentestAssistant()
        
        # Download exploit
        if args.download_exploit:
            logger.info(f"Downloading exploit: {args.download_exploit}")
            exploit_path = assistant.download_exploit(args.download_exploit, args.output_dir)
            if exploit_path:
                logger.success(f"Exploit downloaded to: {exploit_path}")
            else:
                logger.error("Failed to download exploit")
            return
        
        # Search exploits
        if args.search_exploits:
            logger.info(f"Searching ExploitDB for: {args.search_exploits}")
            exploits = assistant.search_exploits(args.search_exploits)
            if exploits:
                logger.success(f"Found {len(exploits)} exploits")
                for i, exploit in enumerate(exploits[:10], 1):
                    logger.info(f"{i}. [{exploit.risk_level.name}] {exploit.title} (ID: {exploit.id})")
            else:
                logger.warning("No exploits found")
            return
        
        # Bloodhound analysis
        if args.bloodhound:
            logger.info(f"Analyzing Bloodhound data: {args.bloodhound}")
            bloodhound_results = await assistant.analyze_bloodhound_data(
                args.bloodhound, 
                args.bloodhound_target
            )
            
            if bloodhound_results:
                logger.success(f"Bloodhound analysis complete: {len(bloodhound_results.get('attack_paths', []))} attack paths found")
                
                # Save Bloodhound report
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f"{args.output_dir}/bloodhound_analysis.json", "w") as f:
                    json.dump(bloodhound_results, f, indent=2)
                
                # Print top paths
                attack_paths = bloodhound_results.get('attack_paths', [])
                if attack_paths:
                    logger.info("Top 3 attack paths:")
                    for i, path in enumerate(attack_paths[:3], 1):
                        logger.info(f"  {i}. Risk: {path.get('risk_score', 0):.1f} - {path.get('description', 'No description')}")
        
        # Run comprehensive scan
        if args.targets and (args.run_tools or args.ad_scan):
            results = await assistant.comprehensive_scan_with_integrations(
                args.targets, 
                ports, 
                credentials
            )
            
            scan_result = results['scan_result']
            
            # Print summary
            logger.success(f"Scan completed!")
            logger.info(f"Hosts found: {len(scan_result.hosts)}")
            logger.info(f"AD hosts: {len(scan_result.ad_hosts)}")
            logger.info(f"Total vulnerabilities: {scan_result.statistics['total_vulnerabilities']}")
            logger.info(f"Total exploits found: {scan_result.statistics['total_exploits']}")
            
            # Show top risky hosts
            risky_hosts = sorted(scan_result.hosts, key=lambda x: x.risk_score, reverse=True)[:3]
            logger.info("Top 3 risky hosts:")
            for host in risky_hosts:
                logger.info(f"  {host.ip} - Risk: {host.risk_score:.1f}")
            
            # Show top exploits
            if scan_result.exploits_found:
                top_exploits = sorted(scan_result.exploits_found, key=lambda x: x.risk_level.value, reverse=True)[:5]
                logger.info("Top 5 exploits:")
                for i, exploit in enumerate(top_exploits, 1):
                    logger.info(f"  {i}. [{exploit.risk_level.name}] {exploit.title}")
        
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
    except Exception as e:
        logger.critical(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())