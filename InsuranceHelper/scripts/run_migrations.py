#!/usr/bin/env python3
"""
Run database migrations using Alembic.

This script runs the Alembic migrations for the database.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_migrations():
    """Run database migrations using Alembic."""
    from alembic.config import Config
    from alembic import command
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Set up the Alembic configuration
    alembic_cfg = Config(str(script_dir.parent / "alembic.ini"))
    
    # Set the script location explicitly
    alembic_cfg.set_main_option("script_location", str(script_dir.parent / "alembic"))
    
    # Run the migrations
    print("Running database migrations...")
    command.upgrade(alembic_cfg, "head")
    print("Migrations completed successfully!")

if __name__ == "__main__":
    run_migrations()
