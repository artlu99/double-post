# Double Post

Interactive CLI tool to reconcile personal CSV records against bank statements. Bank data is treated as source of truth.

## Installation

```bash
# Clone and install
git clone <repo-url> && cd double-post
uv sync  # or: pip install -e .
```

**Requirements**: Python 3.11+, uv (recommended) or pip

## Quick Start

```bash
# Reconcile bank statement against personal records
uv run python -m src.main data/examples/chase_statement.csv data/examples/my_records.csv

# With custom settings
uv run python -m src.main bank.csv personal.csv --min-confidence 0.5 --date-window 5

# Dry run (no TUI)
uv run python -m src.main bank.csv personal.csv --dry-run
```

## How It Works

1. **Load & Normalize**: CSV files are auto-detected and normalized (dates, amounts, descriptions)
2. **Preprocess**:
   - Sign normalization (handles opposite debit/credit conventions)
   - Date filtering (removes records newer than bank statement + 1 day)
   - Reconciled filtering (skips `reconciled=true` records)
3. **Match**: Intelligent matching (first two words + exact amount = 0.90) or fuzzy matching (amount, date, description)
4. **Categorize**:
   - HIGH (≥0.9): Auto-accepted ⭐
   - MEDIUM (0.5-0.9): Manual review ○
   - LOW (0.1-0.5): Weak suggestions ○
5. **Review**: TUI for accept/reject/manual match/edit

## TUI Shortcuts

```
↑↓ or j/k  Navigate    a  Accept    r  Reject    m  Manual match
e  Edit      f  Filter   t  Sort     i  Missing items
u  Unmatched s  Summary  q  Quit
```

## CSV Requirements

**Required columns**: Date, Amount, Description (fuzzy-matched, variations recognized)

**Optional**: `reconciled` column (set `true`/`True` to skip already-matched records)

**Sign conventions**: Auto-detected. For credit cards, assumes debits (purchases) > credits (payments).

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--min-confidence` | 0.1 | Minimum confidence threshold |
| `--date-window` | 3 | Days for date matching |

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines.

## License

MIT
