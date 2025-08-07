#!/usr/bin/env python3
"""
Launcher script for Render deployment.
This ensures the app module can be found regardless of working directory.
"""
import os
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Add the project root to Python path
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Also add it to PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(script_dir) + ':' + os.environ.get('PYTHONPATH', '')

def main():
    """Launch the FastAPI application with uvicorn."""
    try:
        import uvicorn
        from app.main import app
        
        # Get port from environment (Render sets this)
        port = int(os.environ.get('PORT', 8000))
        
        print(f"Starting server on port {port}")
        print(f"Python path: {sys.path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        print(f"Python path: {sys.path}")
        
        # Try to find the app directory
        for root, dirs, files in os.walk('.'):
            if 'app' in dirs:
                print(f"Found 'app' directory in: {root}")
        
        raise
    except Exception as e:
        print(f"Error starting server: {e}")
        raise

if __name__ == "__main__":
    main()
