import sys
import os

# Add the parent directory to sys.path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from amiga.amiga import AMiGA

if __name__ == "__main__":
    AMiGA()