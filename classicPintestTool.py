#!/usr/bin/env python3
"""
🚀 AutoPentest Assistant - ULTIMATE EDITION
Complete penetration testing system without Telegram
Version 4.0 - All-in-One Cyber Weapon
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

# ==================== CONFIGURATION AND DATA TYPES ====================

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
    UNKNOWN = "unknown"

@dataclass
class PortResult:
    port: int
    status: PortStatus
    service: str
    version: str = ""
    banner: str = ""
    cpe: str = ""
    protocol: str = "tcp"

@dataclass
class HostResult:
    ip: str
    hostname: str = ""
    os: str = ""
    ports: List[PortResult] = field(default_factory=list)
    services: Dict[str, Any] = field(default_factory=dict)
    vulnerabilities: List[Any] = field(default_factory=list)
    risk_score: float = 0.0

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
class ScanResult:
    targets: List[str]
    hosts: List[HostResult]
    start_time: datetime
    end_time: datetime
    scan_config: Dict[str, Any]
    statistics: Dict[str, Any] = field(default_factory=dict)

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
                # Try to resolve domain name
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
    def calculate_risk_score(cvss_score: float, exploit_available: bool, service_criticality: float) -> float:
        base_score = cvss_score
        if exploit_available:
            base_score *= 1.3
        return base_score * service_criticality
    
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

# ==================== PARSERS ====================

class NmapParser:
    @staticmethod
    def parse_xml(xml_file: str) -> List[HostResult]:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            hosts = []
            
            for host in root.findall('host'):
                # IP address
                address_elem = host.find('address')
                if address_elem is None:
                    continue
                
                ip = address_elem.get('addr')
                host_result = HostResult(ip=ip)
                
                # Hostname
                hostnames = host.find('hostnames')
                if hostnames:
                    hostname_elem = hostnames.find('hostname')
                    if hostname_elem is not None:
                        host_result.hostname = hostname_elem.get('name')
                
                # Ports
                ports = host.find('ports')
                if ports:
                    for port_elem in ports.findall('port'):
                        port_id = int(port_elem.get('portid'))
                        protocol = port_elem.get('protocol', 'tcp')
                        
                        state_elem = port_elem.find('state')
                        status = PortStatus.CLOSED
                        if state_elem is not None:
                            state = state_elem.get('state', 'closed')
                            status = PortStatus.OPEN if state == 'open' else PortStatus.CLOSED
                        
                        service_elem = port_elem.find('service')
                        service_name = "unknown"
                        version = ""
                        if service_elem is not None:
                            service_name = service_elem.get('name', 'unknown')
                            version = service_elem.get('version', '')
                            version = service_elem.get('product', '') + ' ' + version
                        
                        port_result = PortResult(
                            port=port_id,
                            status=status,
                            service=service_name,
                            version=version.strip(),
                            protocol=protocol
                        )
                        host_result.ports.append(port_result)
                
                hosts.append(host_result)
            
            return hosts
        except Exception as e:
            Logger().error(f"Nmap XML parsing error: {e}")
            return []

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
            if port in [21, 22, 23, 25, 80, 110, 143, 443, 993, 995]:
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
            3389: 'rdp', 5432: 'postgresql', 5900: 'vnc', 6379: 'redis'
        }
        return common_ports.get(port, 'unknown')
    
    async def scan_host(self, ip: str, ports: List[int]) -> List[PortResult]:
        self.logger.info(f"Scanning host {ip} ({len(ports)} ports)")
        
        tasks = [self.scan_port(ip, port) for port in ports]
        results = await asyncio.gather(*tasks)
        
        open_ports = [result for result in results if result is not None]
        self.logger.success(f"Found {len(open_ports)} open ports on {ip}")
        
        return open_ports

# ==================== SERVICE DETECTORS ====================

class HTTPDetector:
    def __init__(self):
        self.session = None
        self.logger = Logger()
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
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
                
                # Find forms
                info['forms'] = self._extract_forms(body)
                
                # Cookies
                info['cookies'] = [str(cookie) for cookie in response.cookies.values()]
                
                return info
                
        except Exception as e:
            self.logger.warning(f"HTTP service detection error {url}: {e}")
            return {'error': str(e)}
    
    def _extract_forms(self, html_content: str) -> List[Dict[str, Any]]:
        forms = []
        form_pattern = r'<form[^>]*>(.*?)</form>'
        
        for form_match in re.finditer(form_pattern, html_content, re.DOTALL | re.IGNORECASE):
            form_html = form_match.group(0)
            form = {
                'action': self._extract_form_action(form_html),
                'method': self._extract_form_method(form_html),
                'inputs': self._extract_form_inputs(form_html)
            }
            forms.append(form)
        
        return forms
    
    def _extract_form_action(self, form_html: str) -> str:
        action_match = re.search(r'action=[\'"]([^\'"]*)[\'"]', form_html, re.IGNORECASE)
        return action_match.group(1) if action_match else ''
    
    def _extract_form_method(self, form_html: str) -> str:
        method_match = re.search(r'method=[\'"]([^\'"]*)[\'"]', form_html, re.IGNORECASE)
        return method_match.group(1).upper() if method_match else 'GET'
    
    def _extract_form_inputs(self, form_html: str) -> List[Dict[str, str]]:
        inputs = []
        input_pattern = r'<input[^>]*>'
        
        for input_match in re.finditer(input_pattern, form_html, re.IGNORECASE):
            input_html = input_match.group(0)
            input_data = {}
            
            for attr in ['name', 'type', 'value', 'placeholder']:
                attr_match = re.search(fr'{attr}=[\'"]([^\'"]*)[\'"]', input_html, re.IGNORECASE)
                if attr_match:
                    input_data[attr] = attr_match.group(1)
            
            if input_data:
                inputs.append(input_data)
        
        return inputs

class SSHDetector:
    async def detect_ssh_service(self, ip: str, port: int) -> Dict[str, Any]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=5.0
            )
            
            # Read SSH banner
            banner = await reader.read(1024)
            banner_text = banner.decode('utf-8', errors='ignore').strip()
            
            info = {
                'banner': banner_text,
                'ssh_version': self._extract_ssh_version(banner_text),
                'supported_auth_methods': [],
                'vulnerabilities': []
            }
            
            writer.close()
            await writer.wait_closed()
            
            # Vulnerability analysis
            info['vulnerabilities'] = self._analyze_ssh_banner(banner_text)
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_ssh_version(self, banner: str) -> str:
        version_match = re.search(r'SSH-(\d+\.\d+-[^\s]+)', banner)
        return version_match.group(1) if version_match else 'Unknown'
    
    def _analyze_ssh_banner(self, banner: str) -> List[Dict[str, Any]]:
        vulnerabilities = []
        
        # Check for outdated SSH versions
        if 'SSH-1.99' in banner or 'SSH-1.5' in banner:
            vulnerabilities.append({
                'type': 'weak_ssh_version',
                'severity': 'medium',
                'description': 'Outdated SSH version detected, vulnerable to attacks'
            })
        
        # Check known vulnerable versions
        vulnerable_versions = ['OpenSSH_7.4', 'OpenSSH_7.5', 'OpenSSH_7.6']
        for vuln_ver in vulnerable_versions:
            if vuln_ver in banner:
                vulnerabilities.append({
                    'type': 'vulnerable_ssh_version',
                    'severity': 'high',
                    'description': f'Vulnerable SSH version detected: {vuln_ver}'
                })
        
        return vulnerabilities

class BannerAnalyzer:
    @staticmethod
    def analyze_banner(banner: str, service: str, port: int) -> List[Dict[str, Any]]:
        findings = []
        banner_lower = banner.lower()
        
        # Common vulnerability patterns
        vulnerability_patterns = {
            'ftp': [
                (r'vsftpd\s+2\.3\.4', 'high', 'VSFTPD 2.3.4 Backdoor Command Execution'),
                (r'proftpd', 'info', 'ProFTPD detected')
            ],
            'smtp': [
                (r'sendmail', 'info', 'Sendmail mail server'),
                (r'postfix', 'info', 'Postfix mail server'),
                (r'exim', 'info', 'Exim mail server')
            ],
            'http': [
                (r'apache', 'info', 'Apache web server'),
                (r'nginx', 'info', 'Nginx web server'),
                (r'iis', 'info', 'Microsoft IIS server')
            ]
        }
        
        # Check patterns for specific service
        if service in vulnerability_patterns:
            for pattern, severity, description in vulnerability_patterns[service]:
                if re.search(pattern, banner_lower):
                    findings.append({
                        'type': 'banner_analysis',
                        'severity': severity,
                        'description': description,
                        'evidence': banner[:100]  # First 100 chars of banner
                    })
        
        # Search for software versions
        version_patterns = [
            (r'(\w+)[/\s-]?v?(\d+\.\d+(?:\.\d+)?)', 'software_version')
        ]
        
        for pattern, finding_type in version_patterns:
            matches = re.finditer(pattern, banner)
            for match in matches:
                software, version = match.groups()
                findings.append({
                    'type': finding_type,
                    'severity': 'info',
                    'description': f'Software detected: {software} version {version}',
                    'software': software,
                    'version': version
                })
        
        return findings

# ==================== SCANNER MANAGER ====================

class ScannerManager:
    def __init__(self):
        self.port_scanner = AsyncPortScanner()
        self.http_detector = HTTPDetector()
        self.ssh_detector = SSHDetector()
        self.banner_analyzer = BannerAnalyzer()
        self.logger = Logger()
    
    async def comprehensive_scan(self, ip: str, ports: List[int] = None) -> HostResult:
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 6379]
        
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
                elif port_result.service == 'ssh':
                    task = self._detect_ssh_service(ip, port_result)
                else:
                    task = self._analyze_generic_banner(ip, port_result)
                service_tasks.append(task)
            
            service_results = await asyncio.gather(*service_tasks)
            
            # Update service information
            for port_result, service_info in zip(open_ports, service_results):
                if service_info:
                    host_result.services[f"{port_result.service}_{port_result.port}"] = service_info
                    
                    # Banner analysis for vulnerabilities
                    if port_result.banner:
                        banner_findings = self.banner_analyzer.analyze_banner(
                            port_result.banner, port_result.service, port_result.port
                        )
                        host_result.vulnerabilities.extend(banner_findings)
            
            # Risk calculation
            host_result.risk_score = self._calculate_host_risk(host_result)
            
            self.logger.success(f"Completed scan for host {ip}. Risk score: {host_result.risk_score:.2f}")
            
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
    
    async def _detect_ssh_service(self, ip: str, port_result: PortResult) -> Dict[str, Any]:
        try:
            return await self.ssh_detector.detect_ssh_service(ip, port_result.port)
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_generic_banner(self, ip: str, port_result: PortResult) -> Dict[str, Any]:
        return {
            'service': port_result.service,
            'port': port_result.port,
            'banner': port_result.banner,
            'analysis': self.banner_analyzer.analyze_banner(
                port_result.banner, port_result.service, port_result.port
            )
        }
    
    def _calculate_host_risk(self, host: HostResult) -> float:
        risk_score = 0.0
        
        # Base risk for open ports
        risk_score += len(host.ports) * 0.1
        
        # Increased risk for specific services
        high_risk_services = ['ssh', 'ftp', 'telnet', 'smtp']
        for port in host.ports:
            if port.service in high_risk_services:
                risk_score += 0.5
            if port.service in ['http', 'https']:
                risk_score += 0.3
        
        # Additional risk for vulnerabilities
        for vuln in host.vulnerabilities:
            severity_weights = {'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}
            risk_score += severity_weights.get(vuln.get('severity', 'low'), 0.5)
        
        return min(risk_score, 10.0)  # Maximum risk 10.0

# ==================== CVE AND VULNERABILITY WORK ====================

class NVDClient:
    def __init__(self, api_key: str = None):
        self.base_url = "https://services.nvd.nist.gov/rest/json/cves/1.0"
        self.api_key = api_key
        self.session = None
        self.logger = Logger()
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=10)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_cpe(self, cpe_string: str) -> List[CVEData]:
        """Search CVE by CPE string"""
        if not self.session:
            async with self:
                return await self.search_cpe(cpe_string)
        
        try:
            params = {
                'cpeMatchString': cpe_string,
                'resultsPerPage': 50
            }
            
            if self.api_key:
                params['apiKey'] = self.api_key
            
            async with self.session.get(f"{self.base_url}", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_cve_response(data)
                else:
                    self.logger.warning(f"NVD API returned status {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"NVD API request error: {e}")
            return []
    
    async def search_cve(self, cve_id: str) -> Optional[CVEData]:
        """Search specific CVE by ID"""
        if not self.session:
            async with self:
                return await self.search_cve(cve_id)
        
        try:
            url = f"{self.base_url}/{cve_id}"
            params = {}
            
            if self.api_key:
                params['apiKey'] = self.api_key
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    cves = self._parse_cve_response(data)
                    return cves[0] if cves else None
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"CVE search error {cve_id}: {e}")
            return None
    
    def _parse_cve_response(self, data: Dict[str, Any]) -> List[CVEData]:
        cves = []
        
        if 'result' not in data or 'CVE_Items' not in data['result']:
            return cves
        
        for item in data['result']['CVE_Items']:
            cve_id = item['cve']['CVE_data_meta']['ID']
            
            # Description
            description = ""
            for desc in item['cve']['description']['description_data']:
                if desc['lang'] == 'en':
                    description = desc['value']
                    break
            
            # CVSS score
            cvss_score = 0.0
            cvss_vector = ""
            
            if 'impact' in item and 'baseMetricV3' in item['impact']:
                cvssv3 = item['impact']['baseMetricV3']['cvssV3']
                cvss_score = cvssv3['baseScore']
                cvss_vector = cvssv3['vectorString']
            elif 'impact' in item and 'baseMetricV2' in item['impact']:
                cvssv2 = item['impact']['baseMetricV2']['cvssV2']
                cvss_score = cvssv2['baseScore']
                cvss_vector = cvssv2['vectorString']
            
            # References
            references = []
            for ref in item['cve']['references']['reference_data']:
                references.append(ref['url'])
            
            # Vulnerable products
            vulnerable_products = []
            if 'configurations' in item:
                for node in item['configurations']['nodes']:
                    for cpe in node.get('cpe_match', []):
                        if cpe['vulnerable']:
                            vulnerable_products.append(cpe['cpe23Uri'])
            
            cve_data = CVEData(
                cve_id=cve_id,
                description=description,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                published_date=item.get('publishedDate', ''),
                last_modified=item.get('lastModifiedDate', ''),
                references=references,
                vulnerable_products=vulnerable_products
            )
            
            cves.append(cve_data)
        
        return cves

class RiskCalculator:
    @staticmethod
    def calculate_vulnerability_risk(cve_data: CVEData, context: Dict[str, Any]) -> RiskLevel:
        cvss_score = cve_data.cvss_score
        
        if cvss_score >= 9.0:
            return RiskLevel.CRITICAL
        elif cvss_score >= 7.0:
            return RiskLevel.HIGH
        elif cvss_score >= 4.0:
            return RiskLevel.MEDIUM
        elif cvss_score >= 2.0:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO
    
    @staticmethod
    def generate_remediation_advice(cve_data: CVEData) -> str:
        cvss_score = cve_data.cvss_score
        
        if cvss_score >= 9.0:
            return "IMMEDIATE UPDATE REQUIRED! Critical level vulnerability. Update software to latest version as soon as possible."
        elif cvss_score >= 7.0:
            return "High priority update. Schedule update in the near future."
        elif cvss_score >= 4.0:
            return "Medium priority. Update during next planned maintenance."
        else:
            return "Low priority. Consider update when possible."

# ==================== ANALYSIS AND PRIORITIZATION ====================

class RiskAnalyzer:
    def __init__(self):
        self.risk_calculator = RiskCalculator()
        self.logger = Logger()
    
    def analyze_host_risks(self, host: HostResult, cve_data: List[CVEData]) -> List[Vulnerability]:
        vulnerabilities = []
        
        for cve in cve_data:
            # Context for risk calculation
            context = {
                'host_ip': host.ip,
                'services': [port.service for port in host.ports],
                'open_ports': [port.port for port in host.ports]
            }
            
            risk_level = self.risk_calculator.calculate_vulnerability_risk(cve, context)
            
            vuln = Vulnerability(
                id=str(uuid.uuid4())[:8],
                name=f"CVE-{cve.cve_id} Vulnerability",
                description=cve.description,
                risk_level=risk_level,
                cvss_score=cve.cvss_score,
                cvss_vector=cve.cvss_vector,
                cve_id=cve.cve_id,
                service="Multiple",
                port=0,
                exploit_available=self._check_exploit_availability(cve.cve_id),
                remediation=self.risk_calculator.generate_remediation_advice(cve),
                references=cve.references[:5]  # First 5 references
            )
            
            vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_exploit_availability(self, cve_id: str) -> bool:
        # Known CVEs with public exploits
        exploitable_cves = {
            'CVE-2021-44228', 'CVE-2021-45046', 'CVE-2021-45105',  # Log4Shell
            'CVE-2017-0144', 'CVE-2017-0145',  # EternalBlue
            'CVE-2019-0708',  # BlueKeep
            'CVE-2021-34527',  # PrintNightmare
            'CVE-2018-7600',  # Drupalgeddon
            'CVE-2017-9841',  # PHPUnit
            'CVE-2019-11510',  # Pulse Secure
            'CVE-2020-1472',  # Zerologon,
        }
        
        return cve_id in exploitable_cves

class AttackPlanner:
    def __init__(self):
        self.logger = Logger()
    
    def generate_attack_plan(self, scan_result: ScanResult) -> Dict[str, Any]:
        plan = {
            'critical_targets': [],
            'recommended_actions': [],
            'metasploit_modules': [],
            'nuclei_templates': [],
            'timeline_estimation': 'Unknown'
        }
        
        # Sort hosts by risk score
        sorted_hosts = sorted(scan_result.hosts, key=lambda x: x.risk_score, reverse=True)
        
        for host in sorted_hosts[:3]:  # Top 3 most risky hosts
            host_plan = self._plan_host_attack(host)
            plan['critical_targets'].append(host_plan)
        
        # Generate general recommendations
        plan['recommended_actions'] = self._generate_recommendations(scan_result)
        plan['metasploit_modules'] = self._suggest_metasploit_modules(scan_result)
        plan['nuclei_templates'] = self._suggest_nuclei_templates(scan_result)
        plan['timeline_estimation'] = self._estimate_timeline(scan_result)
        
        return plan
    
    def _plan_host_attack(self, host: HostResult) -> Dict[str, Any]:
        plan = {
            'target': host.ip,
            'risk_score': host.risk_score,
            'attack_vectors': [],
            'initial_access': [],
            'privilege_escalation': [],
            'lateral_movement': []
        }
        
        # Analyze open ports for attack vectors
        for port in host.ports:
            vector = self._analyze_port_for_attack(port, host)
            if vector:
                plan['attack_vectors'].append(vector)
        
        return plan
    
    def _analyze_port_for_attack(self, port: PortResult, host: HostResult) -> Optional[Dict[str, Any]]:
        attack_vectors = {
            21: {'service': 'ftp', 'techniques': ['Brute Force', 'Anonymous Access']},
            22: {'service': 'ssh', 'techniques': ['Brute Force', 'SSH Key Abuse']},
            23: {'service': 'telnet', 'techniques': ['Brute Force', 'Cleartext Credentials']},
            80: {'service': 'http', 'techniques': ['Web App Attacks', 'SQL Injection', 'XSS']},
            443: {'service': 'https', 'techniques': ['Web App Attacks', 'SSL/TLS Attacks']},
            3389: {'service': 'rdp', 'techniques': ['Brute Force', 'BlueKeep']},
            445: {'service': 'smb', 'techniques': ['EternalBlue', 'SMB Relay']},
            1433: {'service': 'mssql', 'techniques': ['Brute Force', 'SQL Injection']},
            3306: {'service': 'mysql', 'techniques': ['Brute Force', 'SQL Injection']},
            5432: {'service': 'postgresql', 'techniques': ['Brute Force', 'SQL Injection']}
        }
        
        if port.port in attack_vectors:
            vector = attack_vectors[port.port]
            return {
                'port': port.port,
                'service': vector['service'],
                'techniques': vector['techniques'],
                'banner': port.banner[:100] if port.banner else 'No banner'
            }
        
        return None
    
    def _generate_recommendations(self, scan_result: ScanResult) -> List[str]:
        recommendations = []
        services_found = set()
        
        for host in scan_result.hosts:
            for port in host.ports:
                services_found.add(port.service)
        
        if 'ssh' in services_found:
            recommendations.append("Check SSH for weak passwords and disable password authentication if possible")
        if 'ftp' in services_found:
            recommendations.append("Check FTP for anonymous access and use SFTP/FTPS instead of FTP")
        if 'telnet' in services_found:
            recommendations.append("REPLACE Telnet with SSH - Telnet transmits data in clear text")
        if any(port.port == 445 for host in scan_result.hosts for port in host.ports):
            recommendations.append("Check SMB for vulnerabilities like EternalBlue and disable SMBv1")
        
        return recommendations
    
    def _suggest_metasploit_modules(self, scan_result: ScanResult) -> List[str]:
        modules = []
        
        for host in scan_result.hosts:
            for port in host.ports:
                if port.service == 'ssh':
                    modules.append("auxiliary/scanner/ssh/ssh_login")
                elif port.service == 'ftp':
                    modules.append("auxiliary/scanner/ftp/ftp_login")
                elif port.port == 445:
                    modules.append("exploit/windows/smb/ms17_010_eternalblue")
                elif port.service in ['http', 'https']:
                    modules.append("auxiliary/scanner/http/http_version")
                    modules.append("auxiliary/scanner/http/dir_scanner")
        
        return list(set(modules))[:10]  # Unique modules, max 10
    
    def _suggest_nuclei_templates(self, scan_result: ScanResult) -> List[str]:
        templates = []
        
        for host in scan_result.hosts:
            for port in host.ports:
                if port.service in ['http', 'https']:
                    templates.extend([
                        "technologies/tech-detect.yaml",
                        "vulnerabilities/generic/generic-cve.yaml",
                        "exposures/configs/git-config.yaml",
                        "exposures/configs/backup-files.yaml"
                    ])
        
        return list(set(templates))[:10]
    
    def _estimate_timeline(self, scan_result: ScanResult) -> str:
        total_ports = sum(len(host.ports) for host in scan_result.hosts)
        total_hosts = len(scan_result.hosts)
        
        if total_hosts == 0:
            return "No targets for estimation"
        
        # Very simplified time estimation
        base_time = total_hosts * 2  # 2 minutes per host
        port_time = total_ports * 0.5  # 30 seconds per port
        
        total_minutes = base_time + port_time
        
        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            hours = total_minutes / 60
            return f"~{hours:.1f} hours"

class Prioritizer:
    @staticmethod
    def prioritize_vulnerabilities(vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        return sorted(vulnerabilities, key=lambda x: (x.risk_level.value, x.cvss_score), reverse=True)
    
    @staticmethod
    def prioritize_hosts(hosts: List[HostResult]) -> List[HostResult]:
        return sorted(hosts, key=lambda x: x.risk_score, reverse=True)

# ==================== OUTPUT RESULTS ====================

class ConsoleOutput:
    def __init__(self):
        self.logger = Logger()
    
    def print_scan_summary(self, scan_result: ScanResult):
        print(f"\n{Color.CYAN}{Color.BOLD}=== AUTOMATED PENTEST - REPORT ==={Color.END}\n")
        
        print(f"{Color.WHITE}Scan Targets:{Color.END}")
        for target in scan_result.targets:
            print(f"  • {target}")
        
        print(f"\n{Color.WHITE}Statistics:{Color.END}")
        print(f"  • Hosts scanned: {len(scan_result.hosts)}")
        print(f"  • Total open ports: {sum(len(host.ports) for host in scan_result.hosts)}")
        print(f"  • Scan duration: {scan_result.end_time - scan_result.start_time}")
        
        # Top risky hosts
        risky_hosts = Prioritizer.prioritize_hosts(scan_result.hosts)[:5]
        print(f"\n{Color.WHITE}Top 5 Risky Hosts:{Color.END}")
        for i, host in enumerate(risky_hosts, 1):
            risk_color = Color.RED if host.risk_score > 7.0 else Color.YELLOW if host.risk_score > 4.0 else Color.GREEN
            print(f"  {i}. {host.ip} - Risk Score: {risk_color}{host.risk_score:.2f}{Color.END}")
    
    def print_host_details(self, host: HostResult):
        print(f"\n{Color.CYAN}{Color.BOLD}=== HOST DETAILS {host.ip} ==={Color.END}")
        
        if host.hostname:
            print(f"{Color.WHITE}Hostname:{Color.END} {host.hostname}")
        
        print(f"{Color.WHITE}Risk Score:{Color.END} {self._get_risk_color(host.risk_score)}{host.risk_score:.2f}{Color.END}")
        
        if host.ports:
            print(f"\n{Color.WHITE}OPEN PORTS:{Color.END}")
            for port in sorted(host.ports, key=lambda x: x.port):
                status_color = Color.GREEN if port.status == PortStatus.OPEN else Color.YELLOW
                print(f"  {port.port}/tcp - {status_color}{port.status.value}{Color.END} - {port.service} {port.version}")
                if port.banner:
                    print(f"      Banner: {port.banner[:100]}...")
        
        if host.vulnerabilities:
            print(f"\n{Color.WHITE}DETECTED VULNERABILITIES:{Color.END}")
            for vuln in sorted(host.vulnerabilities, key=lambda x: x.get('severity', 'info'), reverse=True):
                severity = vuln.get('severity', 'info')
                severity_color = self._get_severity_color(severity)
                print(f"  [{severity_color}{severity.upper()}{Color.END}] {vuln.get('description', 'Unknown')}")
    
    def print_attack_plan(self, attack_plan: Dict[str, Any]):
        print(f"\n{Color.RED}{Color.BOLD}=== ATTACK PLAN ==={Color.END}")
        
        print(f"\n{Color.WHITE}CRITICAL TARGETS:{Color.END}")
        for target in attack_plan['critical_targets']:
            print(f"  • {target['target']} (risk: {target['risk_score']:.2f})")
            for vector in target['attack_vectors']:
                print(f"    - {vector['service']} on port {vector['port']}: {', '.join(vector['techniques'])}")
        
        print(f"\n{Color.WHITE}RECOMMENDED ACTIONS:{Color.END}")
        for action in attack_plan['recommended_actions']:
            print(f"  • {action}")
        
        print(f"\n{Color.WHITE}METASPLOIT MODULES:{Color.END}")
        for module in attack_plan['metasploit_modules'][:5]:
            print(f"  • {module}")
        
        print(f"\n{Color.WHITE}TIME ESTIMATION:{Color.END} {attack_plan['timeline_estimation']}")
    
    def _get_risk_color(self, score: float) -> str:
        if score >= 7.0:
            return Color.RED
        elif score >= 4.0:
            return Color.YELLOW
        else:
            return Color.GREEN
    
    def _get_severity_color(self, severity: str) -> str:
        colors = {
            'critical': Color.RED,
            'high': Color.RED,
            'medium': Color.YELLOW,
            'low': Color.GREEN,
            'info': Color.BLUE
        }
        return colors.get(severity, Color.WHITE)

class JSONOutput:
    @staticmethod
    def generate_report(scan_result: ScanResult, attack_plan: Dict[str, Any]) -> str:
        report = {
            'metadata': {
                'tool': 'AutoPentest Assistant',
                'version': '4.0',
                'scan_date': scan_result.start_time.isoformat(),
                'duration': str(scan_result.end_time - scan_result.start_time)
            },
            'targets': scan_result.targets,
            'hosts': [asdict(host) for host in scan_result.hosts],
            'statistics': scan_result.statistics,
            'attack_plan': attack_plan,
            'summary': {
                'total_hosts': len(scan_result.hosts),
                'total_ports': sum(len(host.ports) for host in scan_result.hosts),
                'high_risk_hosts': len([h for h in scan_result.hosts if h.risk_score > 7.0]),
                'critical_vulnerabilities': sum(len([v for v in host.vulnerabilities if v.get('severity') == 'critical']) for host in scan_result.hosts)
            }
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)

class HTMLOutput:
    @staticmethod
    def generate_report(scan_result: ScanResult, attack_plan: Dict[str, Any]) -> str:
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>AutoPentest Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .critical { color: #e74c3c; font-weight: bold; }
                .high { color: #e67e22; }
                .medium { color: #f39c12; }
                .low { color: #27ae60; }
                .host { background: #f8f9fa; margin: 10px 0; padding: 10px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 AutoPentest Assistant - Report</h1>
                <p>Generated: {scan_date}</p>
            </div>
            
            <div class="section">
                <h2>📊 Scan Statistics</h2>
                <p>Hosts: {host_count} | Ports: {port_count} | Duration: {duration}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Critical Targets</h2>
                {critical_targets}
            </div>
            
            <div class="section">
                <h2>🔍 Host Details</h2>
                {host_details}
            </div>
            
            <div class="section">
                <h2>⚡ Attack Plan</h2>
                <h3>Recommended Actions:</h3>
                <ul>{recommendations}</ul>
                
                <h3>Metasploit Modules:</h3>
                <ul>{metasploit_modules}</ul>
            </div>
        </body>
        </html>
        """
        
        # Fill template with data
        critical_targets = ""
        for target in attack_plan['critical_targets'][:3]:
            critical_targets += f'<div class="host"><strong>{target["target"]}</strong> (risk: {target["risk_score"]:.2f})</div>'
        
        host_details = ""
        for host in scan_result.hosts[:5]:
            host_details += f"""
            <div class="host">
                <h3>{host.ip}</h3>
                <p>Risk Score: <span class="{HTMLOutput._get_risk_class(host.risk_score)}">{host.risk_score:.2f}</span></p>
                <p>Open Ports: {len(host.ports)}</p>
            </div>
            """
        
        recommendations = "".join(f"<li>{rec}</li>" for rec in attack_plan['recommended_actions'][:5])
        metasploit_modules = "".join(f"<li>{mod}</li>" for mod in attack_plan['metasploit_modules'][:5])
        
        return html_template.format(
            scan_date=scan_result.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            host_count=len(scan_result.hosts),
            port_count=sum(len(host.ports) for host in scan_result.hosts),
            duration=str(scan_result.end_time - scan_result.start_time),
            critical_targets=critical_targets,
            host_details=host_details,
            recommendations=recommendations,
            metasploit_modules=metasploit_modules
        )
    
    @staticmethod
    def _get_risk_class(score: float) -> str:
        if score >= 7.0:
            return "critical"
        elif score >= 4.0:
            return "high"
        elif score >= 2.0:
            return "medium"
        else:
            return "low"

# ==================== EXPLOIT DATABASE INTEGRATION ====================

class ExploitDBIntegration:
    def __init__(self):
        self.base_url = "https://www.exploit-db.com"
        self.local_path = "/usr/share/exploitdb"  # Default ExploitDB path
        self.logger = Logger()
    
    def search_local_exploits(self, query: str, service: str = "", port: int = 0) -> List[Exploit]:
        """Search exploits in local ExploitDB database"""
        exploits = []
        
        try:
            # Search in files_exploits.csv
            exploits_csv = Path(f"{self.local_path}/files_exploits.csv")
            if exploits_csv.exists():
                with open(exploits_csv, 'r', encoding='latin-1') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if self._matches_query(row, query, service, port):
                            exploit = Exploit(
                                id=row['id'],
                                title=row['description'],
                                description=row['description'],
                                author=row['author'],
                                date=row['date'],
                                type=row['type'],
                                platform=row['platform'],
                                port=int(row['port']) if row['port'] else 0,
                                verified=row['verified'] == 'True'
                            )
                            exploits.append(exploit)
            
            self.logger.success(f"Found {len(exploits)} local exploits for query: {query}")
            
        except Exception as e:
            self.logger.error(f"Error searching local exploits: {e}")
        
        return exploits[:20]  # Limit results
    
    def _matches_query(self, row: Dict[str, str], query: str, service: str, port: int) -> bool:
        """Check if exploit matches search criteria"""
        if not query.lower() in row['description'].lower():
            return False
        
        if service and service.lower() not in row['description'].lower():
            return False
            
        if port and row['port'] and int(row['port']) != port:
            return False
            
        return True
    
    async def search_online_exploits(self, query: str) -> List[Exploit]:
        """Search exploits online via ExploitDB API"""
        exploits = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search via exploit-db API
                search_url = f"{self.base_url}/search"
                params = {'q': query}
                
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        exploits = self._parse_exploitdb_html(content)
                    
            self.logger.success(f"Found {len(exploits)} online exploits for: {query}")
            
        except Exception as e:
            self.logger.error(f"Error searching online exploits: {e}")
        
        return exploits
    
    def _parse_exploitdb_html(self, html_content: str) -> List[Exploit]:
        """Parse ExploitDB search results HTML"""
        exploits = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find exploit entries (simplified parser)
        exploit_rows = soup.find_all('tr')[1:]  # Skip header
        
        for row in exploit_rows[:10]:  # Limit to first 10
            try:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    exploit = Exploit(
                        id=cols[0].text.strip(),
                        title=cols[1].text.strip(),
                        description=cols[1].text.strip(),
                        date=cols[2].text.strip(),
                        author=cols[3].text.strip(),
                        type=cols[4].text.strip(),
                        platform=cols[5].text.strip() if len(cols) > 5 else "",
                        port=0
                    )
                    exploits.append(exploit)
            except Exception as e:
                continue
        
        return exploits
    
    def get_exploit_code(self, exploit_id: str) -> Optional[str]:
        """Get exploit source code by ID"""
        try:
            # Try local exploitdb first
            exploits_path = Path(f"{self.local_path}/exploits")
            for ext in ['rb', 'py', 'c', 'cpp', 'php']:
                exploit_file = exploits_path / f"{exploit_id}.{ext}"
                if exploit_file.exists():
                    with open(exploit_file, 'r') as f:
                        return f.read()
            
            # Try online download
            download_url = f"{self.base_url}/download/{exploit_id}"
            response = requests.get(download_url)
            if response.status_code == 200:
                return response.text
                
        except Exception as e:
            self.logger.error(f"Error getting exploit code {exploit_id}: {e}")
        
        return None

# ==================== METASPLOIT INTEGRATION ====================

class MetasploitIntegration:
    def __init__(self):
        self.msf_path = "/usr/share/metasploit-framework"
        self.logger = Logger()
    
    def search_modules(self, query: str, module_type: str = "") -> List[Dict[str, str]]:
        """Search Metasploit modules"""
        modules = []
        
        try:
            cmd = ["msfconsole", "-q", "-x", f"search {query}; exit"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                modules = self._parse_msf_search(result.stdout)
            
            self.logger.success(f"Found {len(modules)} Metasploit modules for: {query}")
            
        except Exception as e:
            self.logger.error(f"Error searching Metasploit modules: {e}")
        
        return modules
    
    def _parse_msf_search(self, output: str) -> List[Dict[str, str]]:
        """Parse Metasploit search output"""
        modules = []
        lines = output.split('\n')
        
        for line in lines:
            if line.startswith('exploit/') or line.startswith('auxiliary/'):
                parts = line.split()
                if len(parts) >= 3:
                    module = {
                        'type': parts[0].split('/')[0],
                        'name': parts[0],
                        'disclosure_date': parts[1] if len(parts) > 1 else '',
                        'rank': parts[2] if len(parts) > 2 else '',
                        'description': ' '.join(parts[3:]) if len(parts) > 3 else ''
                    }
                    modules.append(module)
        
        return modules
    
    def generate_rc_file(self, scan_result: ScanResult, output_file: str = "metasploit_scan.rc"):
        """Generate Metasploit resource file for automated scanning"""
        commands = [
            "# Metasploit Auto-Scan Script",
            "# Generated by AutoPentest Assistant",
            "sleep 2",
            ""
        ]
        
        # Basic settings
        commands.extend([
            "setg THREADS 10",
            "setg TIMEOUT 10",
            ""
        ])
        
        # Host-specific modules
        for host in scan_result.hosts:
            commands.append(f"# Target: {host.ip}")
            
            for port in host.ports:
                if port.service == 'ssh':
                    commands.extend(self._generate_ssh_commands(host.ip, port.port))
                elif port.service in ['http', 'https']:
                    commands.extend(self._generate_http_commands(host.ip, port.port, port.service))
                elif port.service == 'ftp':
                    commands.extend(self._generate_ftp_commands(host.ip, port.port))
                elif port.port == 445:
                    commands.extend(self._generate_smb_commands(host.ip))
            
            commands.append("")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(commands))
        
        self.logger.success(f"Metasploit RC file generated: {output_file}")
        return commands
    
    def _generate_ssh_commands(self, ip: str, port: int) -> List[str]:
        return [
            f"use auxiliary/scanner/ssh/ssh_version",
            f"set RHOSTS {ip}",
            f"set RPORT {port}",
            "run",
            "",
            f"use auxiliary/scanner/ssh/ssh_login",
            f"set RHOSTS {ip}",
            f"set RPORT {port}",
            "set USERNAME root",
            "set PASS_FILE /usr/share/wordlists/rockyou.txt",
            "run",
            ""
        ]
    
    def _generate_http_commands(self, ip: str, port: int, service: str) -> List[str]:
        protocol = "https" if service == "https" else "http"
        return [
            f"use auxiliary/scanner/http/http_version",
            f"set RHOSTS {ip}",
            f"set RPORT {port}",
            "run",
            "",
            f"use auxiliary/scanner/http/dir_scanner",
            f"set RHOSTS {ip}",
            f"set RPORT {port}",
            "run",
            ""
        ]
    
    def _generate_ftp_commands(self, ip: str, port: int) -> List[str]:
        return [
            f"use auxiliary/scanner/ftp/ftp_version",
            f"set RHOSTS {ip}",
            f"set RPORT {port}",
            "run",
            "",
            f"use auxiliary/scanner/ftp/ftp_login",
            f"set RHOSTS {ip}",
            f"set RPORT {port}",
            "run",
            ""
        ]
    
    def _generate_smb_commands(self, ip: str) -> List[str]:
        return [
            f"use auxiliary/scanner/smb/smb_version",
            f"set RHOSTS {ip}",
            "run",
            "",
            f"use auxiliary/scanner/smb/smb_ms17_010",
            f"set RHOSTS {ip}",
            "run",
            ""
        ]

# ==================== NUCLEI INTEGRATION ====================

class NucleiIntegration:
    def __init__(self):
        self.templates_path = "/home/kali/nuclei-templates"
        self.logger = Logger()
    
    def generate_scan_command(self, scan_result: ScanResult) -> str:
        """Generate Nuclei scan command for discovered services"""
        targets = []
        
        for host in scan_result.hosts:
            for port in host.ports:
                if port.service in ['http', 'https']:
                    protocol = 'https' if port.service == 'https' else 'http'
                    targets.append(f"{protocol}://{host.ip}:{port.port}")
        
        if not targets:
            return "# No HTTP/HTTPS targets found for Nuclei"
        
        # Write targets to file
        targets_file = "nuclei_targets.txt"
        with open(targets_file, 'w') as f:
            for target in targets:
                f.write(f"{target}\n")
        
        cmd = (
            f"nuclei -l {targets_file} "
            f"-t {self.templates_path}/technologies/ "
            f"-t {self.templates_path}/vulnerabilities/ "
            f"-t {self.templates_path}/exposures/ "
            f"-t {self.templates_path}/misconfiguration/ "
            f"-o nuclei_scan_results.txt "
            f"-rate-limit 100 "
            f"-timeout 10"
        )
        
        self.logger.success(f"Nuclei targets file generated: {targets_file}")
        return cmd
    
    def get_relevant_templates(self, technologies: List[str]) -> List[str]:
        """Get relevant Nuclei templates based on detected technologies"""
        templates = []
        tech_mapping = {
            'WordPress': ['wordpress/', 'cms/wordpress'],
            'Drupal': ['drupal/', 'cms/drupal'],
            'Joomla': ['joomla/', 'cms/joomla'],
            'Apache': ['apache/', 'technologies/apache'],
            'Nginx': ['nginx/', 'technologies/nginx'],
            'IIS': ['iis/', 'technologies/iis'],
        }
        
        for tech in technologies:
            if tech in tech_mapping:
                templates.extend(tech_mapping[tech])
        
        return list(set(templates))

# ==================== ALL INTEGRATIONS MANAGER ====================

class ToolIntegrationManager:
    def __init__(self):
        self.exploit_db = ExploitDBIntegration()
        self.metasploit = MetasploitIntegration()
        self.nuclei = NucleiIntegration()
        self.logger = Logger()
    
    def generate_all_integrations(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate configurations for all integrated tools"""
        integrations = {}
        
        # ExploitDB
        exploits = []
        for host in scan_result.hosts:
            for port in host.ports:
                service_exploits = self.exploit_db.search_local_exploits(port.service, port.service, port.port)
                exploits.extend(service_exploits)
        integrations['exploit_db'] = exploits[:20]
        
        # Metasploit
        integrations['metasploit'] = self.metasploit.generate_rc_file(scan_result)
        
        # Nuclei
        integrations['nuclei'] = self.nuclei.generate_scan_command(scan_result)
        
        self.logger.success("Generated integrations for all tools")
        return integrations
    
    def save_all_integrations(self, integrations: Dict[str, Any], output_dir: str = "tool_integrations"):
        """Save all integration files to directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Metasploit RC file
        if 'metasploit' in integrations:
            with open(f"{output_dir}/metasploit_scan.rc", 'w') as f:
                f.write('\n'.join(integrations['metasploit']))
        
        # Save tool commands
        tools = ['nuclei']
        for tool in tools:
            if tool in integrations:
                if isinstance(integrations[tool], list):
                    content = '\n'.join(integrations[tool])
                else:
                    content = integrations[tool]
                
                with open(f"{output_dir}/{tool}_commands.txt", 'w') as f:
                    f.write(content)
        
        # Save exploits
        if 'exploit_db' in integrations:
            with open(f"{output_dir}/exploits.json", 'w') as f:
                json.dump([asdict(exp) for exp in integrations['exploit_db']], f, indent=2)
        
        self.logger.success(f"All integration files saved to: {output_dir}")

# ==================== MAIN APPLICATION ====================

class AutoPentestAssistant:
    def __init__(self):
        self.scanner_manager = ScannerManager()
        self.nvd_client = NVDClient()
        self.risk_analyzer = RiskAnalyzer()
        self.attack_planner = AttackPlanner()
        self.tool_manager = ToolIntegrationManager()
        self.console_output = ConsoleOutput()
        self.logger = Logger()
    
    async def comprehensive_scan(self, targets: List[str], ports: List[int] = None) -> ScanResult:
        """Perform comprehensive scanning of targets"""
        start_time = datetime.now()
        self.logger.info(f"Starting comprehensive scan of {len(targets)} targets")
        
        # Expand targets (CIDR, domains, etc.)
        expanded_targets = NetworkUtils.expand_targets(targets)
        self.logger.info(f"Expanded to {len(expanded_targets)} IP addresses")
        
        # Scan each host
        scan_tasks = []
        for target in expanded_targets:
            if NetworkUtils.validate_ip(target):
                task = self.scanner_manager.comprehensive_scan(target, ports)
                scan_tasks.append(task)
        
        hosts = await asyncio.gather(*scan_tasks)
        
        # Filter hosts with results
        valid_hosts = [host for host in hosts if host.ports]
        
        end_time = datetime.now()
        
        scan_result = ScanResult(
            targets=targets,
            hosts=valid_hosts,
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
                'total_open_ports': sum(len(host.ports) for host in valid_hosts),
                'scan_duration': str(end_time - start_time)
            }
        )
        
        return scan_result
    
    async def enhanced_scan_with_cve(self, targets: List[str]) -> ScanResult:
        """Enhanced scanning with CVE analysis"""
        self.logger.info("Starting enhanced scanning with CVE analysis")
        
        # Basic scanning
        scan_result = await self.comprehensive_scan(targets)
        
        # CVE analysis for each host
        async with self.nvd_client as nvd:
            for host in scan_result.hosts:
                self.logger.info(f"Analyzing CVE for host {host.ip}")
                
                # Collect CPE from banners and services
                cpe_list = self._extract_cpe_from_host(host)
                
                # Search CVE for each CPE
                for cpe in cpe_list:
                    cve_data = await nvd.search_cpe(cpe)
                    vulnerabilities = self.risk_analyzer.analyze_host_risks(host, cve_data)
                    host.vulnerabilities.extend(vulnerabilities)
                
                # Update risk score considering CVE
                host.risk_score = self._calculate_enhanced_risk(host)
        
        return scan_result
    
    def _extract_cpe_from_host(self, host: HostResult) -> List[str]:
        """Extract CPE from host information"""
        cpe_list = []
        
        for port in host.ports:
            if port.banner:
                # Try to extract software information from banner
                software_matches = re.findall(r'([A-Za-z]+)[/\s-]?v?(\d+\.\d+(?:\.\d+)?)', port.banner)
                for software, version in software_matches:
                    if software.lower() in ['apache', 'nginx', 'openssh', 'vsftpd', 'proftpd']:
                        cpe = f"cpe:/a:{software.lower()}:{software.lower()}:{version}"
                        cpe_list.append(cpe)
        
        return cpe_list
    
    def _calculate_enhanced_risk(self, host: HostResult) -> float:
        """Calculate enhanced risk score considering CVE"""
        base_risk = host.risk_score
        
        # Increase risk for critical CVE
        for vuln in host.vulnerabilities:
            if hasattr(vuln, 'cvss_score'):
                if vuln.cvss_score >= 9.0:
                    base_risk += 2.0
                elif vuln.cvss_score >= 7.0:
                    base_risk += 1.0
                elif vuln.cvss_score >= 4.0:
                    base_risk += 0.5
        
        return min(base_risk, 10.0)
    
    def generate_reports(self, scan_result: ScanResult, output_formats: List[str] = ['console', 'json']):
        """Generate reports in various formats"""
        attack_plan = self.attack_planner.generate_attack_plan(scan_result)
        
        reports = {}
        
        if 'console' in output_formats:
            self.console_output.print_scan_summary(scan_result)
            for host in Prioritizer.prioritize_hosts(scan_result.hosts)[:3]:
                self.console_output.print_host_details(host)
            self.console_output.print_attack_plan(attack_plan)
        
        if 'json' in output_formats:
            reports['json'] = JSONOutput.generate_report(scan_result, attack_plan)
            with open('scan_report.json', 'w', encoding='utf-8') as f:
                f.write(reports['json'])
            self.logger.success("JSON report saved to scan_report.json")
        
        if 'html' in output_formats:
            reports['html'] = HTMLOutput.generate_report(scan_result, attack_plan)
            with open('scan_report.html', 'w', encoding='utf-8') as f:
                f.write(reports['html'])
            self.logger.success("HTML report saved to scan_report.html")
        
        # Generate integration files
        self._generate_integration_files(scan_result)
        
        return reports
    
    def _generate_integration_files(self, scan_result: ScanResult):
        """Generate files for external tools"""
        # Metasploit
        msf_commands = self.tool_manager.metasploit.generate_rc_file(scan_result)
        with open('metasploit_commands.rc', 'w') as f:
            f.write('\n'.join(msf_commands))
        self.logger.success("Metasploit commands saved to metasploit_commands.rc")
        
        # Nuclei
        nuclei_cmd = self.tool_manager.nuclei.generate_scan_command(scan_result)
        with open('nuclei_command.txt', 'w') as f:
            f.write(nuclei_cmd)
        self.logger.success("Nuclei command saved to nuclei_command.txt")

async def main_async():
    """Async main function to handle await calls"""
    parser = argparse.ArgumentParser(description='🚀 AutoPentest Assistant - Ultimate Penetration Testing Tool')
    parser.add_argument('targets', nargs='+', help='Targets to scan (IP, CIDR, domains)')
    parser.add_argument('-p', '--ports', help='Ports to scan (default: common ports)')
    parser.add_argument('--cve-scan', action='store_true', help='Enable CVE vulnerability analysis')
    parser.add_argument('--exploit-scan', action='store_true', help='Search for exploits in ExploitDB')
    parser.add_argument('--output', choices=['console', 'json', 'html', 'all'], default='console', 
                       help='Output format (default: console)')
    parser.add_argument('--generate-tools', action='store_true', help='Generate tool integration files')
    parser.add_argument('--max-hosts', type=int, default=50, help='Maximum hosts to scan')
    
    args = parser.parse_args()
    
    # Parse ports
    ports = None
    if args.ports:
        if '-' in args.ports:
            start, end = map(int, args.ports.split('-'))
            ports = list(range(start, end + 1))
        else:
            ports = [int(p) for p in args.ports.split(',')]
    else:
        # Default common ports
        ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080, 8443]
    
    # Limit hosts
    targets = args.targets[:args.max_hosts]
    
    logger = Logger()
    logger.info(f"Starting AutoPentest Assistant for {len(targets)} targets")
    
    # Determine output formats
    output_formats = []
    if args.output == 'all':
        output_formats = ['console', 'json', 'html']
    else:
        output_formats = [args.output]
    
    try:
        assistant = AutoPentestAssistant()
        
        # Perform scanning
        if args.cve_scan:
            scan_result = await assistant.enhanced_scan_with_cve(targets)
        else:
            scan_result = await assistant.comprehensive_scan(targets, ports)
        
        # Generate tool integrations if requested
        if args.generate_tools:
            integrations = assistant.tool_manager.generate_all_integrations(scan_result)
            assistant.tool_manager.save_all_integrations(integrations)
        
        # Generate reports
        assistant.generate_reports(scan_result, output_formats)
        
        logger.success("Scanning completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Scanning interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    # Check Python version
    if sys.version_info < (3, 7):
        print("Python 3.7 or higher required")
        sys.exit(1)
    
    # Run async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()