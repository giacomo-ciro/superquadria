"""`python -m harness_3d {generate,play,run,replay}`.

The 2D harness keeps `python main.py ...`; the 3D one gets its own module
entrypoint rather than breaking that command shape.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
