# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

---

## 🔒 Reporting a Vulnerability (Confidential Bug Reporting)

We take the security and integrity of **Veriflux** seriously. Because Veriflux is an evaluation, testing, and guardrail framework for AI/ML pipelines, any security vulnerability or risk is treated with high priority.

To ensure responsible and confidential handling without public exposure:

> [!CAUTION]
> **Do not disclose potential security vulnerabilities in public GitHub issues, discussions, pull requests, or forums.**

### How to Report Privately to Maintainers

All security vulnerability reports are handled exclusively through **GitHub Private Vulnerability Reporting**:

1. Navigate directly to the private submission page:  
   👉 **[Report a Security Advisory on GitHub](https://github.com/leaderrr-shivam/veriflux/security/advisories/new)**
   *(Alternatively, go to the repository's **Security** tab, select **Advisories**, and click **Report a vulnerability**).*
2. Complete the confidential advisory form:
   - **Title**: A clear summary of the issue.
   - **Severity / Ecosystem**: Select affected area (e.g. `veriflux`).
   - **Description**: Detailed description of the vulnerability, reproduction steps, minimal proof-of-concept (PoC), and potential impact.
3. Submit the advisory.

### Who Sees the Report?
- **Only the repository owner (`leaderrr-shivam`) and designated authorized maintainers** can see or triage private vulnerability advisories.
- The report remains strictly private and confidential until a fix is coordinated, tested, and released.

---

## 🛡️ Security Architecture & Design Principles

Veriflux is engineered with defense-in-depth design principles:
- **Zero Dynamic Code Execution:** No usage of `eval()`, `exec()`, or `compile()` on user or model inputs.
- **Safe Serialization:** No usage of Python `pickle`. Internal snapshot caching uses a custom type-tagged binary format designed to prevent arbitrary code execution attacks.
- **Input Sanitization & Normalization:** All text inputs are normalized and sanitized prior to evaluation.
- **Zero Core Supply-Chain Risk:** The core evaluation and testing engine requires zero third-party packages, reducing external dependency vulnerabilities and supply-chain tampering risks.

---

## ⚖️ Scope & Heuristic Tooling Disclaimer

- Veriflux provides algorithmic, statistical, and pattern-based heuristics to assist developers in evaluating AI behavior and screening common vulnerabilities.
- **Heuristic Limitations:** No automated heuristic or guardrail can detect or prevent 100% of adversarial attacks, novel prompt injection patterns, evasion techniques, or model hallucinations.
- Veriflux is provided strictly on an **"AS IS"** basis without warranty of any kind. It should not be deployed as an impenetrable security boundary or as the sole safety mechanism for high-risk, life-critical, or safety-critical production deployments.
