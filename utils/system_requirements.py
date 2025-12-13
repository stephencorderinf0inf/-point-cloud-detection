"""
System Requirements Checker
"""

import platform
import psutil
import sys

def get_cpu_info():
    """Get CPU core count and approximate speed."""
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    return cores, threads

def get_ram_info():
    """Get RAM in GB."""
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    return ram_gb

def check_system_requirements():
    """Check if system meets minimum requirements."""
    
    print("\n" + "=" * 60)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 60)
    
    # Check OS
    os_name = platform.system()
    os_version = platform.version()
    print(f"\n✓ OS: {os_name} {os_version}")
    
    if os_name != "Windows":
        print("  ⚠️  Scanner designed for Windows (may work on Linux/Mac)")
    
    # Check Python version
    py_version = sys.version_info
    print(f"\n✓ Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version < (3, 12):
        print("  ❌ Python 3.12+ required!")
        return False
    elif 'a' in sys.version or 'b' in sys.version:
        print("  ⚠️  Pre-release Python detected (use stable 3.12)")
    
    # Check CPU
    cores, threads = get_cpu_info()
    print(f"\n✓ CPU: {cores} cores, {threads} threads")
    
    if cores < 2:
        print("  ❌ Minimum 2 cores required")
        return False
    elif cores < 4:
        print("  ⚠️  4+ cores recommended for best performance")
    else:
        print("  ✅ Excellent CPU for scanning")
    
    # Check RAM
    ram_gb = get_ram_info()
    print(f"\n✓ RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 4:
        print("  ❌ Minimum 4 GB RAM required")
        return False
    elif ram_gb < 8:
        print("  ⚠️  8 GB RAM recommended")
    else:
        print("  ✅ Sufficient RAM")
    
    # Check disk space
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024 ** 3)
    print(f"\n✓ Storage: {free_gb:.1f} GB free")
    
    if free_gb < 2:
        print("  ❌ Minimum 2 GB free space required")
        return False
    elif free_gb < 10:
        print("  ⚠️  10+ GB recommended for multiple scans")
    else:
        print("  ✅ Sufficient storage")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM MEETS REQUIREMENTS")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    check_system_requirements()