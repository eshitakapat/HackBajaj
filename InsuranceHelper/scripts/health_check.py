#!/usr/bin/env python3
"""
Health check script for the Insurance Helper API.

This script checks if the API is running and responding to requests.
"""
import sys
import requests
from urllib.parse import urljoin

def check_health(base_url: str = "http://localhost:8000"):
    """Check the health of the API.
    
    Args:
        base_url: Base URL of the API (default: http://localhost:8000)
    """
    health_url = urljoin(base_url, "/health")
    
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ API is running and healthy!")
        print(f"   Status: {data.get('status')}")
        print(f"   Version: {data.get('version')}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check failed: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Health check for Insurance Helper API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    if not check_health(args.url):
        sys.exit(1)
