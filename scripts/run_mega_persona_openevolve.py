"""Alias for the canonical MegaPersona OpenEvolve runner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_mega_persona_evolution import main


if __name__ == "__main__":
    main()
