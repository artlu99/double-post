# CLAUDE.md

## Overview

**Double Post** is a CSV reconciliation tool matching personal records against bank statements. Bank data is source of truth.

**Stack**: Python 3.11+, uv, Typer, Textual, pandas, RapidFuzz, pytest

## Architecture

```
src/
├── main.py          # CLI, preprocessing pipeline
├── csv_loader.py    # CSV loading, normalization, column mapping
├── matcher.py       # Matching algorithm, sign normalization
├── models.py        # Dataclasses (Match, MatchResult, etc.)
├── aliases.py       # Merchant alias database (SQLite)
├── stubs/textual/  # Type stubs for Textual framework
└── tui/
    ├── app.py       # Main Textual app
    ├── screens.py   # TUI screens (review, missing, summary, unmatched)
    └── manual_match_screen.py
```

## Coding Standards

- **Strict typing required** - All functions must have type hints
- Use `X | Y` syntax, not `Union[X, Y]`
- Use `X | None`, not `Optional[X]`
- Use `list[X]`, `dict[K, V]`
- `dataclass` for structured data
- Specific exceptions, never bare `except:`
- Pandas: use `.loc[]`/`.iloc[]`, prefer vectorized operations
- **TDD methodology**: Write tests BEFORE implementation

## Module Guidance

### `main.py`
CLI entry point. Orchestrates preprocessing: sign normalization → date filtering → reconciled filtering → matching. Use `typer.echo()`, not `print()`.

### `csv_loader.py`
CSV loading with encoding detection, fuzzy column matching, date/amount/description normalization. Use `Decimal` for money, `dateutil.parser` for dates.

### `matcher.py`
**Intelligent matching** (0.90): First two words + exact amount. **Fuzzy matching**: `amount*0.3 + date*0.3 + description*0.4`. Sign normalization detects conventions by frequency (assumes debits > credits for credit cards). Use RapidFuzz, return `MatchResult`.

### `tui/screens.py`
Textual screens with DataTable widgets. **Cursor sync**: Before accept/reject/manual match actions, call `_sync_cursor_to_selected_idx()` to sync state with table cursor position.

## TDD

**Current**: ~73% coverage (230 tests). **Target**: >80%.

Tests: `test_matcher.py`, `test_csv_loader.py`, `test_sign_normalization.py`, `test_date_filtering.py`, `test_reconciled_filtering.py`, `test_tui_*.py`, `test_models.py`, `test_aliases.py`

**Cycle**: Write test (fails) → Implement (passes) → Refactor

## Quick Reference

```bash
uv add pkg           # Add dependency
uv sync              # Install deps
pytest               # Run tests
pytest --cov=src     # Coverage
ruff format src      # Format
ruff check --fix     # Lint
mypy src/            # Type check
```

## Data Folder

```
data/
├── inputs/   # Your CSVs (not tracked)
├── outputs/  # Reconciled files (not tracked)
└── examples/ # Sample CSVs (tracked)
```

## When Adding Features

1. Write test (verify it fails)
2. Implement minimal code
3. Refactor with tests green
4. Type hints, ruff format, update docs
