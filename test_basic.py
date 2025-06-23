#!/usr/bin/env python3
"""Basic test script"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Test basic imports
    from selvage_eval.config.settings import load_config, get_default_config_path
    print("[SUCCESS] Config import successful")
    
    # Test config loading
    config_path = "configs/selvage-eval-config.yml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"[SUCCESS] Config loaded: {config.agent_model}")
    else:
        print("[WARNING] Config file not found, but import worked")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()