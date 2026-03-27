# CLAUDE.md

## Overview

**Double Post** is a CSV reconciliation tool matching personal records against bank statements. Bank data is source of truth.

**Stack**: Python 3.11+, uv, Typer, Textual, pandas, RapidFuzz, pytest

## Coding Standards

- **Strict typing required** - All functions must have type hints
- `dataclass` for structured data
- Pandas: use `.loc[]`/`.iloc[]`, prefer vectorized operations
- **TDD methodology**: Write tests BEFORE implementation

## Module Guidance

### `main.py`
CLI entry point. Orchestrates preprocessing: sign normalization → date filtering → reconciled filtering → matching. Use `typer.echo()`, not `print()`.

### `csv_loader.py`
CSV loading with encoding detection, fuzzy column matching, date/amount/description normalization. Use `Decimal` for money, `dateutil.parser` for dates.

### `matcher.py`
Two-pass matching: intelligent matching (first two words + exact amount) then fuzzy matching (weighted combination of amount, date, and description similarity). Sign normalization detects conventions by frequency. Use RapidFuzz, return `MatchResult`.

### `tui/screens.py`
Textual screens with DataTable widgets. **Cursor sync**: Before accept/reject actions, call `_sync_cursor_to_selected_idx()` to sync state with table cursor position.

## When Adding Features

**Note**: Use `uv run` for all commands (e.g., `uv run pytest`, `uv run ruff`).

1. Write test (verify it fails)
2. Implement minimal code
3. Refactor with tests green
4. Type hints, ruff format, update docs