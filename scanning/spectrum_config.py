"""
Spectrum Analyzer Configuration
Detects laser dots across different wavelengths (380nm - 780nm visible + IR extensions)
"""

import numpy as np
import cv2

class SpectrumAnalyzer:
    """
    Configurable spectrum analyzer for laser detection.
    Supports wavelengths from 380nm (near-UV) to 1000nm (near-IR).
    """
    
    # Wavelength to HSV mapping (visible spectrum)
    SPECTRUM_MAP = {
        # Near-UV to Violet (380-450nm)
        380: {'name': 'Near-UV', 'hue_range': (130, 145), 'note': 'Camera may not detect well'},
        400: {'name': 'Violet', 'hue_range': (130, 145), 'note': 'Very dim on camera'},
        450: {'name': 'Blue', 'hue_range': (100, 130), 'note': 'Clear detection'},
        
        # Blue to Cyan (450-500nm)
        475: {'name': 'Blue-Cyan', 'hue_range': (90, 110), 'note': 'Good detection'},
        500: {'name': 'Cyan', 'hue_range': (80, 100), 'note': 'Excellent detection'},
        
        # Green (500-570nm) - Camera's peak sensitivity
        520: {'name': 'Green', 'hue_range': (50, 70), 'note': 'BEST detection (camera peak)'},
        532: {'name': 'Green Laser', 'hue_range': (45, 65), 'note': 'Common laser pointer'},
        550: {'name': 'Yellow-Green', 'hue_range': (40, 60), 'note': 'Very bright on camera'},
        
        # Yellow to Orange (570-620nm)
        570: {'name': 'Yellow', 'hue_range': (25, 35), 'note': 'Good detection'},
        590: {'name': 'Orange', 'hue_range': (10, 25), 'note': 'Fair detection'},
        
        # Red (620-750nm)
        620: {'name': 'Red-Orange', 'hue_range': (5, 15), 'note': 'Fair detection'},
        635: {'name': 'Red Laser (Bosch)', 'hue_range': (0, 20), 'note': 'Your current laser'},
        650: {'name': 'Deep Red', 'hue_range': (0, 10), 'note': 'Dimmer on camera'},
        670: {'name': 'Far Red', 'hue_range': (170, 180), 'note': 'Very dim'},
        
        # Near-IR (750-1000nm)
        780: {'name': 'Near-IR', 'hue_range': (170, 180), 'note': 'Camera barely detects'},
        850: {'name': 'IR (security)', 'hue_range': None, 'note': 'Invisible to camera (use IR mode)'},
        940: {'name': 'IR (remote)', 'hue_range': None, 'note': 'Invisible to camera'},
    }
    
    def __init__(self, wavelength_nm=None):
        """
        Initialize spectrum analyzer.
        
        Args:
            wavelength_nm: Specific wavelength in nm (380-1000), or None for full spectrum
        """
        self.wavelength_nm = wavelength_nm
        self.mode = "SPECIFIC" if wavelength_nm else "FULL_SPECTRUM"
        
        if wavelength_nm:
            self.hue_range, self.name, self.note = self._get_wavelength_params(wavelength_nm)
        else:
            self.hue_range = None  # Full spectrum mode
            self.name = "Full Spectrum (380-780nm)"
            self.note = "Detecting all visible wavelengths"
    
    def _get_wavelength_params(self, wavelength):
        """Get HSV parameters for a specific wavelength."""
        
        # Check if exact wavelength is in map
        if wavelength in self.SPECTRUM_MAP:
            data = self.SPECTRUM_MAP[wavelength]
            return data['hue_range'], data['name'], data['note']
        
        # Interpolate between known wavelengths
        wavelengths = sorted(self.SPECTRUM_MAP.keys())
        
        # Find bracketing wavelengths
        lower = max([w for w in wavelengths if w <= wavelength], default=wavelengths[0])
        upper = min([w for w in wavelengths if w >= wavelength], default=wavelengths[-1])
        
        if lower == upper:
            data = self.SPECTRUM_MAP[lower]
            return data['hue_range'], data['name'], data['note']
        
        # Linear interpolation of hue range
        lower_data = self.SPECTRUM_MAP[lower]
        upper_data = self.SPECTRUM_MAP[upper]
        
        if lower_data['hue_range'] is None or upper_data['hue_range'] is None:
            # IR range - no HSV detection possible
            return None, f"{wavelength}nm (IR)", "Not visible to RGB camera"
        
        t = (wavelength - lower) / (upper - lower)
        
        hue_min = int(lower_data['hue_range'][0] + t * (upper_data['hue_range'][0] - lower_data['hue_range'][0]))
        hue_max = int(lower_data['hue_range'][1] + t * (upper_data['hue_range'][1] - lower_data['hue_range'][1]))
        
        name = f"{wavelength}nm ({lower_data['name']}-{upper_data['name']})"
        note = f"Interpolated between {lower}nm and {upper}nm"
        
        return (hue_min, hue_max), name, note
    
    def detect_spectrum_dot(self, frame, brightness_threshold=140, min_area=5, max_area=1000,
                           saturation_min=80, value_min=100):
        """
        Detect laser dot at configured wavelength.
        
        Returns:
            dot_x, dot_y, dot_area, all_dots, combined_mask, color_mask, bright_mask
        """
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness mask (common to all modes)
        _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        if self.mode == "FULL_SPECTRUM":
            # Full spectrum: just use brightness + high saturation
            # (any bright, saturated point is likely a laser)
            lower = np.array([0, saturation_min, value_min])
            upper = np.array([180, 255, 255])
            color_mask = cv2.inRange(hsv, lower, upper)
        
        elif self.hue_range is None:
            # IR mode - no color detection possible
            # Use only brightness (IR lasers appear as bright white spots)
            color_mask = np.ones_like(gray) * 255
            print(f"⚠️  IR wavelength ({self.wavelength_nm}nm) - using brightness-only detection")
        
        else:
            # Specific wavelength mode
            hue_min, hue_max = self.hue_range
            
            # Handle red wraparound (hue 170-180 and 0-10)
            if hue_min > 160 or hue_max < 20:
                # Red range spans 0 degree point
                lower_red1 = np.array([0, saturation_min, value_min])
                upper_red1 = np.array([min(20, hue_max), 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                
                lower_red2 = np.array([max(160, hue_min), saturation_min, value_min])
                upper_red2 = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                # Normal hue range
                lower = np.array([hue_min, saturation_min, value_min])
                upper = np.array([hue_max, 255, 255])
                color_mask = cv2.inRange(hsv, lower, upper)
        
        # Combine color and brightness
        combined_mask = cv2.bitwise_and(color_mask, bright_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_dots = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    brightness = gray[cy, cx]
                    h, s, v = hsv[cy, cx]
                    
                    valid_dots.append({
                        'x': cx,
                        'y': cy,
                        'area': area,
                        'brightness': int(brightness),
                        'hue': int(h),
                        'saturation': int(s),
                        'value': int(v),
                        'wavelength': self._estimate_wavelength(int(h))
                    })
        
        if len(valid_dots) == 0:
            return None, None, None, [], combined_mask, color_mask, bright_mask
        
        # Return brightest dot
        best = max(valid_dots, key=lambda d: float(d['saturation']) * float(d['brightness']))
        return best['x'], best['y'], best['area'], valid_dots, combined_mask, color_mask, bright_mask
    
    def _estimate_wavelength(self, hue):
        """Estimate wavelength from HSV hue value (rough approximation)."""
        
        # Hue to wavelength mapping (approximate)
        # HSV hue range: 0-180 in OpenCV
        if 0 <= hue <= 10 or 170 <= hue <= 180:
            return 635  # Red
        elif 10 < hue <= 25:
            return 590  # Orange
        elif 25 < hue <= 40:
            return 570  # Yellow
        elif 40 < hue <= 80:
            return 520  # Green
        elif 80 < hue <= 110:
            return 480  # Cyan
        elif 110 < hue <= 140:
            return 450  # Blue
        else:
            return 420  # Violet
    
    def get_info_text(self):
        """Get info text for display."""
        if self.mode == "FULL_SPECTRUM":
            return f"Mode: {self.name} | {self.note}"
        else:
            if self.hue_range:
                hue_text = f"Hue: {self.hue_range[0]}-{self.hue_range[1]}"
            else:
                hue_text = "IR Detection (brightness only)"
            return f"Mode: {self.name} | {self.wavelength_nm}nm | {hue_text} | {self.note}"
    
    @staticmethod
    def show_spectrum_guide():
        """Print spectrum detection guide."""
        print("\n" + "=" * 80)
        print("🌈 SPECTRUM ANALYZER - WAVELENGTH GUIDE")
        print("=" * 80)
        print("\nVisible Spectrum (380nm - 780nm):")
        print("  380-450nm  Violet/Blue    - Dim on camera, hue 130-145")
        print("  450-500nm  Blue/Cyan      - Clear, hue 80-130")
        print("  500-570nm  Green          - BRIGHTEST on camera! hue 40-70")
        print("  532nm      Green Laser    - Common laser pointer (excellent)")
        print("  570-590nm  Yellow/Orange  - Good, hue 10-40")
        print("  590-650nm  Orange/Red     - Fair, hue 0-25")
        print("  635nm      Red Laser      - Bosch GLM42 (your laser)")
        print("  650-750nm  Deep Red       - Dim, hue 0-10, 170-180")
        print("\nNear-IR (invisible to eye, may appear on camera):")
        print("  780-1000nm IR             - Very dim/invisible, brightness only")
        print("\nCamera Sensitivity:")
        print("  BEST:   500-550nm (green) - Camera sensor peaks here")
        print("  GOOD:   450-600nm (blue to orange)")
        print("  FAIR:   600-700nm (red)")
        print("  POOR:   <450nm, >700nm (violet, deep red, IR)")
        print("\n💡 TIP: For best results, use green lasers (520-532nm)")
        print("=" * 80)