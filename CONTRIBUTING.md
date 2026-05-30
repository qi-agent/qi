# Contributing

PRs welcome. By contributing you agree to the ISC license terms.

## Development

```sh
uv sync
```

## Quality gates

Run in order before submitting:

```sh
uv run ruff check src/
uv run mypy src/
uv run pytest
```

All three must pass.

## Testing

TDD expected. Tests go in `tests/` alongside the module structure. Run with:

```sh
uv run pytest
```

Coverage is tracked but not enforced by a minimum threshold.
