"""Add the project root to sys.path so that validator, config, etc. are importable."""
import sys
import os

# Insert the project root (one level up from this tests/ directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
