"""`python -m engine {generate,play,run,replay}`. See README.md."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
