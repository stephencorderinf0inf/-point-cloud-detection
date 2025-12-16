"""
i18n Integration Example
========================
Shows how to add internationalization to your scanner.

This is a reference implementation - use this pattern in laser_3d_scanner_advanced.py
"""

# Step 1: Import i18n at the top of your file
from i18n_manager import setup_i18n, set_language, get_language, _

# Step 2: Initialize in main function
def main():
    """Main scanner function with i18n support."""
    
    # Initialize i18n (auto-detects system language)
    _ = setup_i18n()  # This MUST be called before using _()
    
    print("=" * 70)
    print(_("3D LASER SCANNER"))
    print(_("Language: {}").format(get_language().upper()))
    print("=" * 70)
    
    # Example: Status messages
    print(_("Initializing camera..."))
    print(_("Loading calibration..."))
    print(_("Camera ready"))
    
    # Example: Mode names
    modes = {
        'laser': _("RED LASER BEAM"),
        'curve': _("CURVE TRACING"),
        'corners': _("CORNER DETECTION"),
        'depth': _("AI DEPTH")
    }
    
    print("\n" + _("Available modes:"))
    for key, name in modes.items():
        print(f"  {key}: {name}")
    
    # Example: Error handling
    try:
        # Simulated error
        raise FileNotFoundError("calibration.npz")
    except FileNotFoundError as e:
        print(_("Error: Calibration file not found"))
        print(_("Please run calibration first"))
    
    # Example: Success messages with formatting
    point_count = 1250
    print(_("Scan complete: {} points captured").format(point_count))
    
    # Example: Controls help
    print("\n" + _("CONTROLS:"))
    controls = [
        ("1", _("Toggle mode")),
        ("SPACE", _("Capture point")),
        ("s", _("Save point cloud")),
        ("m", _("Mesh method")),
        ("q", _("Quit")),
    ]
    
    for key, description in controls:
        print(f"  {key:8s} - {description}")
    
    # Example: Language switching (for testing)
    print("\n" + "=" * 70)
    print(_("LANGUAGE DEMO"))
    print("=" * 70)
    
    demo_message = "Calibration loaded successfully"
    
    print(f"\nEnglish: {demo_message}")
    
    set_language('es')
    print(f"Spanish: {_(demo_message)}")
    
    # Switch back to system default
    set_language('en')
    
    print("\n" + _("Scanner ready!"))


# Step 3: How to use in OpenCV windows
def opencv_window_example():
    """Example of using i18n with OpenCV windows."""
    import cv2
    import numpy as np
    
    # Initialize i18n
    _ = setup_i18n()
    
    # Create a sample frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add translated text to frame
    cv2.putText(frame, _("Calibration Required"), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(frame, _("Press SPACE to capture"), (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Window title (note: OpenCV doesn't support Unicode well)
    cv2.imshow("Scanner - " + _("Preview"), frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


# Step 4: File paths and technical terms (DON'T translate)
def what_not_to_translate():
    """Examples of what should NOT be wrapped with _()"""
    
    # DON'T translate:
    log_file = "scanner_log.txt"  # File paths
    DEBUG = True  # Constants
    mode_id = "MODE_LASER"  # Technical IDs
    
    # Log messages (optional - usually not translated)
    print(f"[DEBUG] Camera initialized at {time.time()}")
    
    # DO translate user-facing messages
    print(_("Camera initialized"))


if __name__ == "__main__":
    # Test the examples
    main()
    
    print("\n" + "=" * 70)
    print("OpenCV Window Example")
    print("=" * 70)
    opencv_window_example()
    
    print("\n" + "=" * 70)
    print("✓ i18n integration examples complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Add 'from i18n_manager import setup_i18n, _' to scanner")
    print("2. Call setup_i18n() at the start of main function")
    print("3. Wrap all user-facing strings with _()")
    print("4. Run extract_translations.py to generate .po files")
    print("5. Translate and compile")
