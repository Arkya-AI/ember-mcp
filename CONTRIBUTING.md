# Contributing to Ember MCP

Thanks for your interest in contributing. Ember is a local-first MCP memory server — contributions that preserve the privacy-first, zero-cloud design are most welcome.

## Getting Started

```bash
git clone https://github.com/TimoLabsAI/ember-mcp
cd ember-mcp
pip install -e ".[dev]"
```

## What to Work On

Check [open issues](https://github.com/TimoLabsAI/ember-mcp/issues) for tagged bugs and feature requests. Issues tagged `good first issue` are a good starting point.

Before opening a PR for a new feature, open an issue first to discuss the approach — especially for changes to the HESTIA scoring, Shadow-Decay framework, or storage schema.

## Code Style

- Python 3.10+ compatibility required
- Use `async`/`await` throughout — all tool functions are async
- New tools must use FastMCP `@mcp.tool()` decorator with `annotations` dict
- All tool functions must return `str`
- Validate user inputs and return descriptive error strings (not silent coercion)

## Running Tests

```bash
pytest tests/
```

## Pull Request Guidelines

1. One logical change per PR
2. Include a short description of what changed and why
3. Update `CHANGELOG.md` under `[Unreleased]`
4. Do not break backward compatibility for existing `~/.ember/` data — all new `Ember` model fields must have Pydantic defaults

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md) — do not file a public issue.
