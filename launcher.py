"""
Infinity Gems 3D Scanner Launcher
Simple launcher that runs the scanner through embedded Python
"""
import subprocess
import sys
from pathlib import Path

def main():
    # Get the directory where this launcher is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        launcher_dir = Path(sys.executable).parent
    else:
        # Running as script
        launcher_dir = Path(__file__).parent
    
    # Path to embedded Python (only used when running as script)
    python_exe = launcher_dir / "python" / "python.exe"
    
    # Path to GUI launcher
    gui_launcher = launcher_dir / "app" / "launcher_gui.py"
    
    # Optional: Play splash video if exists
    splash_video = launcher_dir / "app" / "splash.mp4"
    if splash_video.exists():
        splash_script = launcher_dir / "app" / "splash_screen.py"
        if splash_script.exists() and python_exe.exists():
            try:
                subprocess.run([str(python_exe), str(splash_script)])
            except:
                pass  # Skip splash if error
    
    # Launch the GUI launcher
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe - import and run directly
        sys.path.insert(0, str(launcher_dir / "app"))
        import launcher_gui
        app = launcher_gui.LauncherGUI()
        app.run()
    else:
        # Running as script - use subprocess
        subprocess.run([str(python_exe), str(gui_launcher)])

if __name__ == "__main__":
    main()
