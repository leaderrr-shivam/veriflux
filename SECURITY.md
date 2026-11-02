# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

---

## Reporting a Vulnerability

The security and integrity of **Veriflux** is paramount. Because Veriflux is designed to act as an evaluation and safety guardrail for production AI applications, any vulnerability (such as a prompt injection bypass or PII leakage) is treated with the highest severity.

If you believe you have discovered a security vulnerability in Veriflux:
1. **Do not disclose the vulnerability publicly.**
2. Send an email with a detailed description, reproduction steps, and proof of concept to **security@veriflux.dev**.
3. We will acknowledge receipt within 24 hours and provide an estimated timeline for remediation.

---

## Security Architecture Guarantees

Veriflux is engineered with defense-in-depth principles:
- **No `eval()`, `exec()`, or `compile()`:** Zero dynamic code execution on user or model inputs.
- **No `pickle`:** Serialization uses a custom type-tagged binary format immune to arbitrary code execution attacks.
- **Input Sanitization:** All text inputs are normalized and sanitized before parsing.
- **Zero Supply-Chain Risk:** The core engine requires zero third-party packages, eliminating dependency-confusion and upstream supply chain attacks.

