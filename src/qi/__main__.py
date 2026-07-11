"""Allow running as `python -m qi`."""

import sys

from .cli import main

sys.exit(main())
