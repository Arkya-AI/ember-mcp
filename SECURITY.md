# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

Please do **not** file a public GitHub issue for security vulnerabilities.

Report security issues by emailing the maintainers at the address listed on [ember.timolabs.dev](https://ember.timolabs.dev). Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any suggested mitigations you have identified

You will receive a response within 72 hours. If the issue is confirmed, a patch will be released as soon as possible and you will be credited in the changelog.

## Scope

Ember MCP runs entirely locally. There is no cloud backend, no network ingress, and no authentication layer. The primary attack surface is:

- **Local file access via `source_path`:** Embers can store file paths that are read back during `ember_deep_recall`. As of v0.2.0, all source paths are restricted to the user's home directory (`Path.home()`). Paths outside this boundary are silently skipped.
- **Embedding model downloads:** The `all-MiniLM-L6-v2` model is downloaded from Hugging Face on first run via `sentence-transformers`. Verify the model hash if operating in a high-trust environment.
- **Local storage at `~/.ember/`:** No encryption at rest. Do not store highly sensitive secrets directly as ember content in shared-machine environments.
