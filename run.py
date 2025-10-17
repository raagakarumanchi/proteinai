#!/usr/bin/env python3
"""
FoldAI - One-Click Viral Demo
Run this to predict protein structures instantly!
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    print("🧬 Welcome to FoldAI!")
    print("⚡ Predicting protein structures in seconds...")
    print()
    main()
