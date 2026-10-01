"""``python -m visu_predict`` - same as the ``visu-predict`` command."""

import sys

from .cli import main

sys.exit(main())
