"""Main execution entry point when invoked via python -m veriflux."""
from __future__ import annotations

import sys
from veriflux.cli.main import cli_entry

if __name__ == "__main__":
    sys.exit(cli_entry())
