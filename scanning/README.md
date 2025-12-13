\# 🎯 3D Laser Scanner - User Guide



\## Quick Start Guide for Best Results



\### 📋 Table of Contents

1\. \[System Requirements](#system-requirements)

2\. \[Setup \& Installation](#setup--installation)

3\. \[Backdrop Recommendations](#backdrop-recommendations)

4\. \[Optimal Scanning Distance](#optimal-scanning-distance)

5\. \[Scanning Workflow](#scanning-workflow)

6\. \[Camera Modes](#camera-modes)

7\. \[Detection Modes](#detection-modes)

8\. \[ROI (Region of Interest)](#roi-region-of-interest)

9\. \[Auto-Capture Feature](#auto-capture-feature)

10\. \[Troubleshooting](#troubleshooting)

11\. \[File Outputs](#file-outputs)



---



\## System Requirements



\### Minimum Requirements

\- \*\*OS:\*\* Windows 10/11, Linux, macOS

\- \*\*Camera:\*\* USB webcam (720p minimum, 1080p recommended)

\- \*\*RAM:\*\* 4GB minimum (8GB recommended)

\- \*\*Python:\*\* 3.8 or higher

\- \*\*GPU:\*\* Optional (speeds up processing)



\### Required Hardware

\- \*\*Laser:\*\* 635nm red laser (Bosch GLM 42 or similar)

&nbsp; - Other wavelengths: 450-650nm supported

&nbsp; - Adjustable via spectrum analyzer



\### Check Your System

```bash

python laser\_3d\_scanner\_advanced.py

\# System check runs automatically on first launch

```



---



\## Setup \& Installation



\### 1. Install Dependencies

```bash

pip install opencv-python numpy pathlib

```



\### 2. Camera Calibration (REQUIRED)

```bash

\# Run calibration first (one-time setup)

cd dual\_checkerboard\_3d

python checkerboard.py



\# Follow on-screen instructions:

\# - Print checkerboard pattern (provided)

\# - Capture 20-30 images from different angles

\# - Save calibration file

```



\*\*⚠️ Scanner will NOT work without calibration!\*\*



\### 3. Launch Scanner

```bash

python laser\_3d\_scanner\_advanced.py



\# Follow project setup prompts:

\# 1. Create new project (recommended)

\# 2. Enter project name

\# 3. Select category

\# 4. Add optional notes

```



---



\## Backdrop Recommendations



\### 🌟 Best Backdrops (Ranked)



\#### 1. \*\*Matte Black\*\* ⭐⭐⭐⭐⭐

```

✅ Absorbs light (no reflections)

✅ Minimal edge detection

✅ Laser doesn't reflect

✅ Clean 3D captures



Materials:

\- Black poster board ($3-5)

\- Black felt fabric ($5-10)

\- Photography backdrop ($15-30)

```



\#### 2. \*\*Uniform Contrasting Color\*\* ⭐⭐⭐⭐

```

✅ Easy object separation

✅ Works with ROI cropping



Examples:

\- White object → Black backdrop

\- Black object → White backdrop

\- Red object → Blue/green backdrop

```



\#### 3. \*\*Matte White\*\* ⭐⭐⭐

```

⚠️  Can reflect laser

⚠️  Shows wrinkles/shadows

✅ Works with ROI



Best for: Laser mode only

```



\#### 4. ❌ \*\*AVOID These Backdrops\*\*

```

❌ Textured surfaces (carpet, wood grain)

❌ Patterned wallpaper

❌ Reflective surfaces (glass, metal)

❌ Fabric with wrinkles

❌ Busy backgrounds



Why: Creates thousands of unwanted edge points!

```



\### 💡 DIY Backdrop Setup ($5)

```

Materials:

1\. 2-3 black poster boards ($3)

2\. Tape to join seamlessly ($1)

3\. Optional: Black felt for smooth finish ($2)



Setup:

1\. Tape boards together (no gaps)

2\. Position 30-50cm behind object

3\. Ensure smooth surface (no wrinkles)

4\. Use indirect lighting (not harsh)

```



---



\## Optimal Scanning Distance



\### 📏 Distance Guidelines



| Object Size | Distance | Expected Density | Notes |

|-------------|----------|------------------|-------|

| \*\*Small\*\* (5-15cm) | 30-50cm | 50-100 pts/cm³ | Get close but not too close |

| \*\*Medium\*\* (15-50cm) | 50-80cm | 20-50 pts/cm³ | Sweet spot for most objects |

| \*\*Large\*\* (50-150cm) | 80-150cm | 5-20 pts/cm³ | Room-scale scanning |

| \*\*Very Large\*\* (>150cm) | 100-300cm | 0.1-1 pt/cm³ | Walls, furniture |



\### 🎯 Finding Your Camera's Sweet Spot



\*\*Test Your Minimum Focus Distance:\*\*

```

1\. Launch scanner

2\. Hold printed text at arm's length

3\. Move slowly toward camera

4\. Stop when text becomes blurry

5\. Move back 5-10cm

6\. This is your minimum focus distance!



Rule: Scan objects 10-20cm beyond this point

```



\### ⚠️ Distance Problems



\*\*Too Close (< 30cm):\*\*

```

❌ Image blurs

❌ Laser line breaks up

❌ Detection fails

❌ Curves disappear

```



\*\*Too Far (> 150cm):\*\*

```

⚠️  Lower point density

⚠️  Less detail captured

⚠️  Larger file sizes needed

✅ Still works for room mapping

```



---



\## Scanning Workflow



\### 🎬 Complete 3D Object Capture



\*\*Recommended Process (10-15 minutes per object):\*\*



```

1\. SETUP (2 min)

&nbsp;  ├─ Position object on platform

&nbsp;  ├─ Set up backdrop 30-50cm behind

&nbsp;  ├─ Check lighting (diffused, no harsh shadows)

&nbsp;  └─ Launch scanner, select Mode 2 (Curves)



2\. CAMERA SETUP (1 min)

&nbsp;  ├─ Press 'v' to enable Cartoon Mode

&nbsp;  ├─ Position camera 40-60cm from object

&nbsp;  ├─ Check focus is sharp

&nbsp;  └─ Click \& drag ROI around object



3\. CAPTURE FRONT (1 min)

&nbsp;  ├─ Ensure ROI is tight around object

&nbsp;  ├─ Press SPACE (auto 3x capture)

&nbsp;  ├─ Wait for "READY" message

&nbsp;  └─ Check point count increased



4\. ROTATE \& CAPTURE (8-12 min)

&nbsp;  ├─ Rotate object 30-45 degrees

&nbsp;  ├─ Adjust ROI if needed (drag new box)

&nbsp;  ├─ Press SPACE (auto 3x capture)

&nbsp;  ├─ Repeat 8-12 times for 360° coverage

&nbsp;  └─ Target: 40,000-60,000 points



5\. SAVE \& EXPORT (1 min)

&nbsp;  ├─ Press 's' to save

&nbsp;  ├─ Files saved to project folder

&nbsp;  ├─ Import .obj into Blender/MeshLab

&nbsp;  └─ View/edit your 3D model!

```



\### 📊 Point Density Goals



| Coverage | Points | Quality | Use Case |

|----------|--------|---------|----------|

| Quick scan | 5,000-10,000 | Basic | Object recognition |

| Standard | 20,000-40,000 | Good | 3D printing prep |

| Detailed | 40,000-80,000 | Excellent | Professional modeling |

| High-res | 100,000+ | Studio | Museum/archival |



---



\## Camera Modes



\### 📷 Default Mode

```

Settings:

\- Brightness: 128 (auto)

\- Exposure: -5 (auto)

\- Contrast: 128

\- Sharpness: 128

\- Saturation: 128



Best for: Laser detection, natural colors

```



\### 🎨 Cartoon Mode (Press 'v')

```

Settings:

\- Brightness: 129

\- Exposure: -8 (reduced motion blur)

\- Contrast: 138 (HIGH - better edges)

\- Sharpness: 160 (HIGH - clear features)

\- Saturation: 178 (HIGH - vivid detection)



Best for: Curve detection, corner detection

✅ Recommended for 3D modeling!



Benefits:

✓ Enhanced edge visibility

✓ Sharper corner detection

✓ Better curve tracing

✓ Reduced noise

```



\*\*Toggle:\*\* Press `v` anytime during scanning



---



\## Detection Modes



\### Mode 1: Laser Detection (Press '1')

```

🔴 635nm Red-Orange Laser (Bosch GLM 42)



Use When:

✓ Precise point-by-point scanning

✓ Measuring specific features

✓ High-accuracy required



Adjustments:

\- i/k: Brightness threshold

\- h/n: Saturation min

\- m/,: Value min

\- r: Reset to defaults



Expected: 1-3 points per capture

```



\### Mode 2: Curve Tracing (Press '2') ⭐ RECOMMENDED

```

📐 Edge \& Curve Detection



Use When:

✓ 3D modeling objects

✓ Capturing complex shapes

✓ Fast area coverage



Features:

\- Auto-detects object edges

\- Traces curves automatically

\- Captures 500-2000 pts per capture



Best Results:

1\. Enable Cartoon Mode ('v')

2\. Use ROI (click \& drag)

3\. Auto-capture (SPACE)

4\. Rotate object frequently

```



\### Mode 3: Corner Detection (Press '3')

```

📍 Geometric Feature Points



Use When:

✓ Box-like objects

✓ Architectural features

✓ Sharp angles needed



Expected: 50-200 points per capture

Works great with Cartoon Mode!

```



---



\## ROI (Region of Interest)



\### 🎯 Why Use ROI?



\*\*Without ROI:\*\*

```

Detected:

\- Object edges ✓

\- Backdrop wrinkles ✗

\- Wall corners ✗

\- Floor edges ✗

\- Shadows ✗

\- Other objects ✗



Result: 50% unwanted points!

```



\*\*With ROI:\*\*

```

Detected:

\- Object edges ✓

\- \[Everything else ignored]



Result: 100% object points! ✅

```



\### 📦 How to Use ROI



```

1\. Position object in frame

2\. Left-click \& drag rectangle around object

3\. Release mouse button

4\. ROI activated! (yellow border)

5\. Press SPACE to capture

6\. Adjust ROI by drawing new rectangle

7\. Press 'x' to clear ROI (full frame)

```



\### 💡 ROI Tips



```

✅ Draw tight around object (10-20px padding)

✅ Adjust ROI when rotating object

✅ Smaller ROI = faster processing

✅ Use with Cartoon Mode for best results



❌ Don't include backdrop edges

❌ Don't include shadows

❌ Don't include other objects

```



---



\## Auto-Capture Feature



\### 🎬 What is Auto-Capture?



\*\*Press SPACE once → 3 automatic captures with 1-second intervals\*\*



\### Why 3 Captures?



```

Capture 1: Object at position A

&nbsp;  ↓ 1 second delay

Capture 2: Object at A + micro-movement

&nbsp;  ↓ 1 second delay  

Capture 3: Object at A + more variation



Result: Better coverage, reduced artifacts

```



\### 📊 On-Screen Countdown



```

Press SPACE →

&nbsp; "CAPTURING 1/3 in 1.0s" (countdown appears)

&nbsp; "CAPTURING 1/3 in 0.5s"

&nbsp; "CAPTURING NOW!" (brief flash)

&nbsp; ✓ Captured!

&nbsp; 

&nbsp; Repeat for 2/3 and 3/3

&nbsp; 

&nbsp; "CAPTURE COMPLETE!"

&nbsp; "Total Points: 5,168"

&nbsp; "🔄 READY - Rotate object"

```



\### 🔄 Toggle Auto-Capture



```

Press 'a' to toggle:

\- AUTO (3x): Press SPACE = 3 captures \[DEFAULT]

\- SINGLE: Press SPACE = 1 capture (legacy)



Indicator shows in top-right corner

```



\### 🎯 Workflow with Auto-Capture



```

1\. Position object, draw ROI

2\. Press SPACE once

3\. Wait 3 seconds (automatic)

4\. See "READY" message

5\. Rotate object 30-45°

6\. Press SPACE again

7\. Repeat 8-12 times

8\. Complete 360° coverage

9\. Press 's' to save

```



---



\## Troubleshooting



\### ❌ "Detection curves go away when too close"



\*\*Problem:\*\* Camera out of focus, detection fails



\*\*Solutions:\*\*

```

1\. Move back 10-20cm

2\. Check if text/edges are sharp

3\. Find your camera's minimum focus distance

4\. Scan objects 40-60cm away (sweet spot)

```



\### ❌ "Too many background points captured"



\*\*Problem:\*\* Backdrop has texture/edges



\*\*Solutions:\*\*

```

1\. Use matte black backdrop

2\. Enable ROI (click \& drag around object)

3\. Draw tighter ROI box

4\. Remove wrinkles from backdrop

5\. Move object away from backdrop

```



\### ❌ "Low point density / sparse scan"



\*\*Problem:\*\* Distance too far, or room-scale capture



\*\*Solutions:\*\*

```

1\. Get closer to object (40-60cm)

2\. Use Cartoon Mode ('v')

3\. Enable ROI for focused capture

4\. Press SPACE more frequently

5\. Make more rotations (12-15 angles)

```



\### ❌ "Laser not detected"



\*\*Problem:\*\* Wrong wavelength, settings off



\*\*Solutions:\*\*

```

1\. Press 'r' to reset to defaults

2\. Press 'w' to change wavelength

3\. Adjust brightness: i/k keys

4\. Check laser is pointing at object

5\. Ensure room isn't too bright

```



\### ❌ "Image is blurry"



\*\*Problem:\*\* Too close, motion blur, or exposure



\*\*Solutions:\*\*

```

1\. Move camera back 10-20cm

2\. Enable Cartoon Mode (-8 exposure)

3\. Hold camera steadier (use tripod)

4\. Ensure object isn't moving

```



\### ❌ "Scanner crashes / freezes"



\*\*Problem:\*\* System resources low



\*\*Solutions:\*\*

```

1\. Close other programs

2\. Disable AI panel ('p' key)

3\. Reduce capture frequency

4\. Check Task Manager for CPU/RAM

5\. Restart scanner

```



---



\## File Outputs



\### 📁 Project Structure



```

projects/

└── your\_project\_name\_20251208\_123456/

&nbsp;   ├── metadata.json          # Project info

&nbsp;   ├── point\_clouds/

&nbsp;   │   ├── scan\_3d\_bosch\_glm42.npz   # Raw point data

&nbsp;   │   └── scan\_3d\_bosch\_glm42.obj   # 3D mesh (import-ready)

&nbsp;   └── session\_report.html    # Quality analysis



ai\_analysis\_results/

└── session\_20251208\_093016/

&nbsp;   ├── ai\_results.csv         # Frame analysis

&nbsp;   ├── session\_summary.json   # Stats

&nbsp;   └── session\_report.html    # Visual report

```



\### 📊 File Formats



\#### .npz (NumPy Archive)

```

Usage: Python/scientific analysis

Contains: Raw XYZ coordinates

Size: 2-5 MB per 100k points



Load in Python:

data = np.load("scan.npz")

points = data\['points']  # Shape: (N, 3)

```



\#### .obj (Wavefront OBJ)

```

Usage: 3D modeling software

Contains: Vertex positions

Import into:

\- Blender (free)

\- MeshLab (free)

\- CloudCompare (free)

\- Maya, 3ds Max, etc.



Format:

v x y z

v x y z

...

```



\#### session\_report.html

```

Usage: Quality analysis

Contains:

\- Frame-by-frame quality graphs

\- Sharpness metrics

\- Brightness analysis

\- FPS performance

\- Overall quality score



Open in web browser to view

```



---



\## Keyboard Shortcuts Reference



\### 🎛️ General Controls

```

SPACE  - Auto-capture 3 points (1 sec intervals)

a      - Toggle auto-capture mode (3x vs single)

s      - Save point cloud \& mesh

c      - Clear all points

q      - Quit scanner

```



\### 📷 Camera Controls

```

v      - Toggle Cartoon Mode (high contrast/sharpness)

p      - Toggle AI analysis panel

b      - Toggle info box

d      - Toggle debug view

```



\### 🎯 Mode Selection

```

1      - Laser detection (635nm)

2      - Curve tracing (recommended for 3D)

3      - Corner detection

4      - Ellipse detection

5      - Cylinder detection

```



\### ✂️ ROI (Region of Interest)

```

LEFT-CLICK \& DRAG - Draw ROI rectangle

x                 - Clear ROI (use full frame)

z                 - Auto-suggest ROI (finds edges)

t                 - Test backdrop quality

```



\### 🔴 Laser Adjustments (Mode 1)

```

i/k    - Brightness threshold (UP/DOWN)

h/n    - Saturation min (DECREASE/INCREASE)

m/,    - Value min (DECREASE/INCREASE)

u/U    - Hue min (DECREASE/INCREASE)

o/O    - Hue max (UP/DOWN)

j/l    - Min area (DECREASE/INCREASE)

\[/]    - Max area (DECREASE/INCREASE)

r      - Reset to defaults

w      - Change wavelength

```



\### 📐 Curve Adjustments (Mode 2/3)

```

e      - Decrease edge sensitivity (fewer curves)

E      - Increase edge sensitivity (more curves)

```



---



\## Best Practices Summary



\### ✅ DO:

```

✓ Calibrate camera before first use

✓ Use matte black backdrop

✓ Enable Cartoon Mode for curve/corner detection

✓ Use ROI to isolate object

✓ Scan at 40-60cm distance

✓ Rotate object 30-45° between captures

✓ Press SPACE 8-12 times for 360° coverage

✓ Keep object steady during auto-capture

✓ Check "READY" message before rotating

```



\### ❌ DON'T:

```

✗ Skip camera calibration

✗ Use textured/patterned backdrops

✗ Scan too close (<30cm) or too far (>150cm)

✗ Move object during countdown

✗ Include backdrop in ROI

✗ Expect high density from room-scale scans

✗ Forget to save before quitting

```



---



\## Example Workflows



\### 📦 Scanning a Small Object (5-15cm)



```bash

\# 1. Setup

python laser\_3d\_scanner\_advanced.py

\# Create project: "Figurine Scan"



\# 2. In scanner:

Press '2'        # Curve mode

Press 'v'        # Cartoon mode ON

Draw ROI         # Click \& drag around object

Press SPACE      # Capture front (auto 3x)



\# 3. Rotate \& capture (repeat 10-12 times):

Rotate 30°

Press SPACE

Wait for "READY"

Repeat...



\# 4. Save

Press 's'        # Saves to project folder



\# Expected result:

\# 40,000-60,000 points

\# 50-100 points/cm³

\# High detail capture

```



\### 🪑 Scanning Medium Object (30-80cm)



```bash

\# Setup distance: 60-80cm

\# Backdrop: Black poster board 40cm behind

\# Lighting: Diffused LED panel



\# Process:

1\. Mode 2 (curves)

2\. Cartoon Mode ON

3\. ROI around object

4\. Rotate 45° each capture

5\. 8 captures = 360°

6\. Save



\# Expected: 20,000-40,000 points

```



\### 🏠 Scanning Room/Large Area



```bash

\# Setup distance: 100-200cm

\# No ROI needed (full frame)

\# Mode: Curves or Corners



\# Process:

1\. Capture from center

2\. Move to different positions

3\. 8-12 captures from different angles

4\. Save



\# Expected: 100,000+ points

\# Density: 0.1-1 point/cm³ (normal for room scale)

```



---



\## Support \& Resources



\### 📚 Documentation

\- `README.md` - This file

\- `CALIBRATION\_GUIDE.md` - Camera calibration steps

\- `API\_REFERENCE.md` - Python API documentation



\### 🐛 Troubleshooting

\- Check `session\_report.html` for quality issues

\- Review `ai\_results.csv` for frame-by-frame analysis

\- Enable debug view ('d' key) to see detection masks



\### 💬 Community

\- GitHub Issues: Report bugs/request features

\- Discussions: Ask questions, share results



---



\## Version History



\### v2.0 (Current)

\- ✅ Auto-capture feature (3x with countdown)

\- ✅ Cartoon mode for enhanced edges

\- ✅ ROI cropping

\- ✅ Spectrum analyzer (380-1000nm)

\- ✅ Project management system

\- ✅ AI quality analysis

\- ✅ GPU acceleration support



\### v1.0

\- Basic laser detection

\- Manual capture

\- Single detection mode



---



\## Credits



\*\*Developed by:\*\* Graphene Scanner Team  

\*\*Camera Calibration:\*\* OpenCV dual-checkerboard method  

\*\*Supported Lasers:\*\* 635nm (Bosch GLM 42) and others  

\*\*License:\*\* MIT



---



\## Quick Reference Card



```

═══════════════════════════════════════════════════════════

&nbsp;             3D SCANNER QUICK REFERENCE

═══════════════════════════════════════════════════════════



SETUP:

&nbsp; 1. Calibrate camera (one-time)

&nbsp; 2. Black backdrop 30-50cm behind object

&nbsp; 3. Camera 40-60cm from object

&nbsp; 4. Diffused lighting



WORKFLOW:

&nbsp; Press '2' → Curve mode

&nbsp; Press 'v' → Cartoon mode ON

&nbsp; Draw ROI → Click \& drag

&nbsp; Press SPACE → Auto-capture 3x

&nbsp; Rotate 30-45° → Repeat 8-12 times

&nbsp; Press 's' → Save



SHORTCUTS:

&nbsp; SPACE   Auto-capture    v  Cartoon mode

&nbsp; 2       Curve mode      s  Save

&nbsp; ROI     Click \& drag    q  Quit



TARGET: 40,000-60,000 points for detailed model

═══════════════════════════════════════════════════════════

```



---



\*\*Ready to scan? Launch the scanner and start capturing!\*\* 🎯✨# 🎯 3D Laser Scanner - User Guide



\## Quick Start Guide for Best Results



\### 📋 Table of Contents

1\. \[System Requirements](#system-requirements)

2\. \[Setup \& Installation](#setup--installation)

3\. \[Backdrop Recommendations](#backdrop-recommendations)

4\. \[Optimal Scanning Distance](#optimal-scanning-distance)

5\. \[Scanning Workflow](#scanning-workflow)

6\. \[Camera Modes](#camera-modes)

7\. \[Detection Modes](#detection-modes)

8\. \[ROI (Region of Interest)](#roi-region-of-interest)

9\. \[Auto-Capture Feature](#auto-capture-feature)

10\. \[Troubleshooting](#troubleshooting)

11\. \[File Outputs](#file-outputs)



---



\## System Requirements



\### Minimum Requirements

\- \*\*OS:\*\* Windows 10/11, Linux, macOS

\- \*\*Camera:\*\* USB webcam (720p minimum, 1080p recommended)

\- \*\*RAM:\*\* 4GB minimum (8GB recommended)

\- \*\*Python:\*\* 3.8 or higher

\- \*\*GPU:\*\* Optional (speeds up processing)



\### Required Hardware

\- \*\*Laser:\*\* 635nm red laser (Bosch GLM 42 or similar)

&nbsp; - Other wavelengths: 450-650nm supported

&nbsp; - Adjustable via spectrum analyzer



\### Check Your System

```bash

python laser\_3d\_scanner\_advanced.py

\# System check runs automatically on first launch

```



---



\## Setup \& Installation



\### 1. Install Dependencies

```bash

pip install opencv-python numpy pathlib

```



\### 2. Camera Calibration (REQUIRED)

```bash

\# Run calibration first (one-time setup)

cd dual\_checkerboard\_3d

python checkerboard.py



\# Follow on-screen instructions:

\# - Print checkerboard pattern (provided)

\# - Capture 20-30 images from different angles

\# - Save calibration file

```



\*\*⚠️ Scanner will NOT work without calibration!\*\*



\### 3. Launch Scanner

```bash

python laser\_3d\_scanner\_advanced.py



\# Follow project setup prompts:

\# 1. Create new project (recommended)

\# 2. Enter project name

\# 3. Select category

\# 4. Add optional notes

```



---



\## Backdrop Recommendations



\### 🌟 Best Backdrops (Ranked)



\#### 1. \*\*Matte Black\*\* ⭐⭐⭐⭐⭐

```

✅ Absorbs light (no reflections)

✅ Minimal edge detection

✅ Laser doesn't reflect

✅ Clean 3D captures



Materials:

\- Black poster board ($3-5)

\- Black felt fabric ($5-10)

\- Photography backdrop ($15-30)

```



\#### 2. \*\*Uniform Contrasting Color\*\* ⭐⭐⭐⭐

```

✅ Easy object separation

✅ Works with ROI cropping



Examples:

\- White object → Black backdrop

\- Black object → White backdrop

\- Red object → Blue/green backdrop

```



\#### 3. \*\*Matte White\*\* ⭐⭐⭐

```

⚠️  Can reflect laser

⚠️  Shows wrinkles/shadows

✅ Works with ROI



Best for: Laser mode only

```



\#### 4. ❌ \*\*AVOID These Backdrops\*\*

```

❌ Textured surfaces (carpet, wood grain)

❌ Patterned wallpaper

❌ Reflective surfaces (glass, metal)

❌ Fabric with wrinkles

❌ Busy backgrounds



Why: Creates thousands of unwanted edge points!

```



\### 💡 DIY Backdrop Setup ($5)

```

Materials:

1\. 2-3 black poster boards ($3)

2\. Tape to join seamlessly ($1)

3\. Optional: Black felt for smooth finish ($2)



Setup:

1\. Tape boards together (no gaps)

2\. Position 30-50cm behind object

3\. Ensure smooth surface (no wrinkles)

4\. Use indirect lighting (not harsh)

```



---



\## Optimal Scanning Distance



\### 📏 Distance Guidelines



| Object Size | Distance | Expected Density | Notes |

|-------------|----------|------------------|-------|

| \*\*Small\*\* (5-15cm) | 30-50cm | 50-100 pts/cm³ | Get close but not too close |

| \*\*Medium\*\* (15-50cm) | 50-80cm | 20-50 pts/cm³ | Sweet spot for most objects |

| \*\*Large\*\* (50-150cm) | 80-150cm | 5-20 pts/cm³ | Room-scale scanning |

| \*\*Very Large\*\* (>150cm) | 100-300cm | 0.1-1 pt/cm³ | Walls, furniture |



\### 🎯 Finding Your Camera's Sweet Spot



\*\*Test Your Minimum Focus Distance:\*\*

```

1\. Launch scanner

2\. Hold printed text at arm's length

3\. Move slowly toward camera

4\. Stop when text becomes blurry

5\. Move back 5-10cm

6\. This is your minimum focus distance!



Rule: Scan objects 10-20cm beyond this point

```



\### ⚠️ Distance Problems



\*\*Too Close (< 30cm):\*\*

```

❌ Image blurs

❌ Laser line breaks up

❌ Detection fails

❌ Curves disappear

```



\*\*Too Far (> 150cm):\*\*

```

⚠️  Lower point density

⚠️  Less detail captured

⚠️  Larger file sizes needed

✅ Still works for room mapping

```



---



\## Scanning Workflow



\### 🎬 Complete 3D Object Capture



\*\*Recommended Process (10-15 minutes per object):\*\*



```

1\. SETUP (2 min)

&nbsp;  ├─ Position object on platform

&nbsp;  ├─ Set up backdrop 30-50cm behind

&nbsp;  ├─ Check lighting (diffused, no harsh shadows)

&nbsp;  └─ Launch scanner, select Mode 2 (Curves)



2\. CAMERA SETUP (1 min)

&nbsp;  ├─ Press 'v' to enable Cartoon Mode

&nbsp;  ├─ Position camera 40-60cm from object

&nbsp;  ├─ Check focus is sharp

&nbsp;  └─ Click \& drag ROI around object



3\. CAPTURE FRONT (1 min)

&nbsp;  ├─ Ensure ROI is tight around object

&nbsp;  ├─ Press SPACE (auto 3x capture)

&nbsp;  ├─ Wait for "READY" message

&nbsp;  └─ Check point count increased



4\. ROTATE \& CAPTURE (8-12 min)

&nbsp;  ├─ Rotate object 30-45 degrees

&nbsp;  ├─ Adjust ROI if needed (drag new box)

&nbsp;  ├─ Press SPACE (auto 3x capture)

&nbsp;  ├─ Repeat 8-12 times for 360° coverage

&nbsp;  └─ Target: 40,000-60,000 points



5\. SAVE \& EXPORT (1 min)

&nbsp;  ├─ Press 's' to save

&nbsp;  ├─ Files saved to project folder

&nbsp;  ├─ Import .obj into Blender/MeshLab

&nbsp;  └─ View/edit your 3D model!

```



\### 📊 Point Density Goals



| Coverage | Points | Quality | Use Case |

|----------|--------|---------|----------|

| Quick scan | 5,000-10,000 | Basic | Object recognition |

| Standard | 20,000-40,000 | Good | 3D printing prep |

| Detailed | 40,000-80,000 | Excellent | Professional modeling |

| High-res | 100,000+ | Studio | Museum/archival |



---



\## Camera Modes



\### 📷 Default Mode

```

Settings:

\- Brightness: 128 (auto)

\- Exposure: -5 (auto)

\- Contrast: 128

\- Sharpness: 128

\- Saturation: 128



Best for: Laser detection, natural colors

```



\### 🎨 Cartoon Mode (Press 'v')

```

Settings:

\- Brightness: 129

\- Exposure: -8 (reduced motion blur)

\- Contrast: 138 (HIGH - better edges)

\- Sharpness: 160 (HIGH - clear features)

\- Saturation: 178 (HIGH - vivid detection)



Best for: Curve detection, corner detection

✅ Recommended for 3D modeling!



Benefits:

✓ Enhanced edge visibility

✓ Sharper corner detection

✓ Better curve tracing

✓ Reduced noise

```



\*\*Toggle:\*\* Press `v` anytime during scanning



---



\## Detection Modes



\### Mode 1: Laser Detection (Press '1')

```

🔴 635nm Red-Orange Laser (Bosch GLM 42)



Use When:

✓ Precise point-by-point scanning

✓ Measuring specific features

✓ High-accuracy required



Adjustments:

\- i/k: Brightness threshold

\- h/n: Saturation min

\- m/,: Value min

\- r: Reset to defaults



Expected: 1-3 points per capture

```



\### Mode 2: Curve Tracing (Press '2') ⭐ RECOMMENDED

```

📐 Edge \& Curve Detection



Use When:

✓ 3D modeling objects

✓ Capturing complex shapes

✓ Fast area coverage



Features:

\- Auto-detects object edges

\- Traces curves automatically

\- Captures 500-2000 pts per capture



Best Results:

1\. Enable Cartoon Mode ('v')

2\. Use ROI (click \& drag)

3\. Auto-capture (SPACE)

4\. Rotate object frequently

```



\### Mode 3: Corner Detection (Press '3')

```

📍 Geometric Feature Points



Use When:

✓ Box-like objects

✓ Architectural features

✓ Sharp angles needed



Expected: 50-200 points per capture

Works great with Cartoon Mode!

```



---



\## ROI (Region of Interest)



\### 🎯 Why Use ROI?



\*\*Without ROI:\*\*

```

Detected:

\- Object edges ✓

\- Backdrop wrinkles ✗

\- Wall corners ✗

\- Floor edges ✗

\- Shadows ✗

\- Other objects ✗



Result: 50% unwanted points!

```



\*\*With ROI:\*\*

```

Detected:

\- Object edges ✓

\- \[Everything else ignored]



Result: 100% object points! ✅

```



\### 📦 How to Use ROI



```

1\. Position object in frame

2\. Left-click \& drag rectangle around object

3\. Release mouse button

4\. ROI activated! (yellow border)

5\. Press SPACE to capture

6\. Adjust ROI by drawing new rectangle

7\. Press 'x' to clear ROI (full frame)

```



\### 💡 ROI Tips



```

✅ Draw tight around object (10-20px padding)

✅ Adjust ROI when rotating object

✅ Smaller ROI = faster processing

✅ Use with Cartoon Mode for best results



❌ Don't include backdrop edges

❌ Don't include shadows

❌ Don't include other objects

```



---



\## Auto-Capture Feature



\### 🎬 What is Auto-Capture?



\*\*Press SPACE once → 3 automatic captures with 1-second intervals\*\*



\### Why 3 Captures?



```

Capture 1: Object at position A

&nbsp;  ↓ 1 second delay

Capture 2: Object at A + micro-movement

&nbsp;  ↓ 1 second delay  

Capture 3: Object at A + more variation



Result: Better coverage, reduced artifacts

```



\### 📊 On-Screen Countdown



```

Press SPACE →

&nbsp; "CAPTURING 1/3 in 1.0s" (countdown appears)

&nbsp; "CAPTURING 1/3 in 0.5s"

&nbsp; "CAPTURING NOW!" (brief flash)

&nbsp; ✓ Captured!

&nbsp; 

&nbsp; Repeat for 2/3 and 3/3

&nbsp; 

&nbsp; "CAPTURE COMPLETE!"

&nbsp; "Total Points: 5,168"

&nbsp; "🔄 READY - Rotate object"

```



\### 🔄 Toggle Auto-Capture



```

Press 'a' to toggle:

\- AUTO (3x): Press SPACE = 3 captures \[DEFAULT]

\- SINGLE: Press SPACE = 1 capture (legacy)



Indicator shows in top-right corner

```



\### 🎯 Workflow with Auto-Capture



```

1\. Position object, draw ROI

2\. Press SPACE once

3\. Wait 3 seconds (automatic)

4\. See "READY" message

5\. Rotate object 30-45°

6\. Press SPACE again

7\. Repeat 8-12 times

8\. Complete 360° coverage

9\. Press 's' to save

```



---



\## Troubleshooting



\### ❌ "Detection curves go away when too close"



\*\*Problem:\*\* Camera out of focus, detection fails



\*\*Solutions:\*\*

```

1\. Move back 10-20cm

2\. Check if text/edges are sharp

3\. Find your camera's minimum focus distance

4\. Scan objects 40-60cm away (sweet spot)

```



\### ❌ "Too many background points captured"



\*\*Problem:\*\* Backdrop has texture/edges



\*\*Solutions:\*\*

```

1\. Use matte black backdrop

2\. Enable ROI (click \& drag around object)

3\. Draw tighter ROI box

4\. Remove wrinkles from backdrop

5\. Move object away from backdrop

```



\### ❌ "Low point density / sparse scan"



\*\*Problem:\*\* Distance too far, or room-scale capture



\*\*Solutions:\*\*

```

1\. Get closer to object (40-60cm)

2\. Use Cartoon Mode ('v')

3\. Enable ROI for focused capture

4\. Press SPACE more frequently

5\. Make more rotations (12-15 angles)

```



\### ❌ "Laser not detected"



\*\*Problem:\*\* Wrong wavelength, settings off



\*\*Solutions:\*\*

```

1\. Press 'r' to reset to defaults

2\. Press 'w' to change wavelength

3\. Adjust brightness: i/k keys

4\. Check laser is pointing at object

5\. Ensure room isn't too bright

```



\### ❌ "Image is blurry"



\*\*Problem:\*\* Too close, motion blur, or exposure



\*\*Solutions:\*\*

```

1\. Move camera back 10-20cm

2\. Enable Cartoon Mode (-8 exposure)

3\. Hold camera steadier (use tripod)

4\. Ensure object isn't moving

```



\### ❌ "Scanner crashes / freezes"



\*\*Problem:\*\* System resources low



\*\*Solutions:\*\*

```

1\. Close other programs

2\. Disable AI panel ('p' key)

3\. Reduce capture frequency

4\. Check Task Manager for CPU/RAM

5\. Restart scanner

```



---



\## File Outputs



\### 📁 Project Structure



```

projects/

└── your\_project\_name\_20251208\_123456/

&nbsp;   ├── metadata.json          # Project info

&nbsp;   ├── point\_clouds/

&nbsp;   │   ├── scan\_3d\_bosch\_glm42.npz   # Raw point data

&nbsp;   │   └── scan\_3d\_bosch\_glm42.obj   # 3D mesh (import-ready)

&nbsp;   └── session\_report.html    # Quality analysis



ai\_analysis\_results/

└── session\_20251208\_093016/

&nbsp;   ├── ai\_results.csv         # Frame analysis

&nbsp;   ├── session\_summary.json   # Stats

&nbsp;   └── session\_report.html    # Visual report

```



\### 📊 File Formats



\#### .npz (NumPy Archive)

```

Usage: Python/scientific analysis

Contains: Raw XYZ coordinates

Size: 2-5 MB per 100k points



Load in Python:

data = np.load("scan.npz")

points = data\['points']  # Shape: (N, 3)

```



\#### .obj (Wavefront OBJ)

```

Usage: 3D modeling software

Contains: Vertex positions

Import into:

\- Blender (free)

\- MeshLab (free)

\- CloudCompare (free)

\- Maya, 3ds Max, etc.



Format:

v x y z

v x y z

...

```



\#### session\_report.html

```

Usage: Quality analysis

Contains:

\- Frame-by-frame quality graphs

\- Sharpness metrics

\- Brightness analysis

\- FPS performance

\- Overall quality score



Open in web browser to view

```



---



\## Keyboard Shortcuts Reference



\### 🎛️ General Controls

```

SPACE  - Auto-capture 3 points (1 sec intervals)

a      - Toggle auto-capture mode (3x vs single)

s      - Save point cloud \& mesh

c      - Clear all points

q      - Quit scanner

```



\### 📷 Camera Controls

```

v      - Toggle Cartoon Mode (high contrast/sharpness)

p      - Toggle AI analysis panel

b      - Toggle info box

d      - Toggle debug view

```



\### 🎯 Mode Selection

```

1      - Laser detection (635nm)

2      - Curve tracing (recommended for 3D)

3      - Corner detection

4      - Ellipse detection

5      - Cylinder detection

```



\### ✂️ ROI (Region of Interest)

```

LEFT-CLICK \& DRAG - Draw ROI rectangle

x                 - Clear ROI (use full frame)

z                 - Auto-suggest ROI (finds edges)

t                 - Test backdrop quality

```



\### 🔴 Laser Adjustments (Mode 1)

```

i/k    - Brightness threshold (UP/DOWN)

h/n    - Saturation min (DECREASE/INCREASE)

m/,    - Value min (DECREASE/INCREASE)

u/U    - Hue min (DECREASE/INCREASE)

o/O    - Hue max (UP/DOWN)

j/l    - Min area (DECREASE/INCREASE)

\[/]    - Max area (DECREASE/INCREASE)

r      - Reset to defaults

w      - Change wavelength

```



\### 📐 Curve Adjustments (Mode 2/3)

```

e      - Decrease edge sensitivity (fewer curves)

E      - Increase edge sensitivity (more curves)

```



---



\## Best Practices Summary



\### ✅ DO:

```

✓ Calibrate camera before first use

✓ Use matte black backdrop

✓ Enable Cartoon Mode for curve/corner detection

✓ Use ROI to isolate object

✓ Scan at 40-60cm distance

✓ Rotate object 30-45° between captures

✓ Press SPACE 8-12 times for 360° coverage

✓ Keep object steady during auto-capture

✓ Check "READY" message before rotating

```



\### ❌ DON'T:

```

✗ Skip camera calibration

✗ Use textured/patterned backdrops

✗ Scan too close (<30cm) or too far (>150cm)

✗ Move object during countdown

✗ Include backdrop in ROI

✗ Expect high density from room-scale scans

✗ Forget to save before quitting

```



---



\## Example Workflows



\### 📦 Scanning a Small Object (5-15cm)



```bash

\# 1. Setup

python laser\_3d\_scanner\_advanced.py

\# Create project: "Figurine Scan"



\# 2. In scanner:

Press '2'        # Curve mode

Press 'v'        # Cartoon mode ON

Draw ROI         # Click \& drag around object

Press SPACE      # Capture front (auto 3x)



\# 3. Rotate \& capture (repeat 10-12 times):

Rotate 30°

Press SPACE

Wait for "READY"

Repeat...



\# 4. Save

Press 's'        # Saves to project folder



\# Expected result:

\# 40,000-60,000 points

\# 50-100 points/cm³

\# High detail capture

```



\### 🪑 Scanning Medium Object (30-80cm)



```bash

\# Setup distance: 60-80cm

\# Backdrop: Black poster board 40cm behind

\# Lighting: Diffused LED panel



\# Process:

1\. Mode 2 (curves)

2\. Cartoon Mode ON

3\. ROI around object

4\. Rotate 45° each capture

5\. 8 captures = 360°

6\. Save



\# Expected: 20,000-40,000 points

```



\### 🏠 Scanning Room/Large Area



```bash

\# Setup distance: 100-200cm

\# No ROI needed (full frame)

\# Mode: Curves or Corners



\# Process:

1\. Capture from center

2\. Move to different positions

3\. 8-12 captures from different angles

4\. Save



\# Expected: 100,000+ points

\# Density: 0.1-1 point/cm³ (normal for room scale)

```



---



\## Support \& Resources



\### 📚 Documentation

\- `README.md` - This file

\- `CALIBRATION\_GUIDE.md` - Camera calibration steps

\- `API\_REFERENCE.md` - Python API documentation



\### 🐛 Troubleshooting

\- Check `session\_report.html` for quality issues

\- Review `ai\_results.csv` for frame-by-frame analysis

\- Enable debug view ('d' key) to see detection masks



\### 💬 Community

\- GitHub Issues: Report bugs/request features

\- Discussions: Ask questions, share results



---



\## Version History



\### v2.0 (Current)

\- ✅ Auto-capture feature (3x with countdown)

\- ✅ Cartoon mode for enhanced edges

\- ✅ ROI cropping

\- ✅ Spectrum analyzer (380-1000nm)

\- ✅ Project management system

\- ✅ AI quality analysis

\- ✅ GPU acceleration support



\### v1.0

\- Basic laser detection

\- Manual capture

\- Single detection mode



---



\## Credits



\*\*Developed by:\*\* Graphene Scanner Team  

\*\*Camera Calibration:\*\* OpenCV dual-checkerboard method  

\*\*Supported Lasers:\*\* 635nm (Bosch GLM 42) and others  

\*\*License:\*\* MIT



---



\## Quick Reference Card



```

═══════════════════════════════════════════════════════════

&nbsp;             3D SCANNER QUICK REFERENCE

═══════════════════════════════════════════════════════════



SETUP:

&nbsp; 1. Calibrate camera (one-time)

&nbsp; 2. Black backdrop 30-50cm behind object

&nbsp; 3. Camera 40-60cm from object

&nbsp; 4. Diffused lighting



WORKFLOW:

&nbsp; Press '2' → Curve mode

&nbsp; Press 'v' → Cartoon mode ON

&nbsp; Draw ROI → Click \& drag

&nbsp; Press SPACE → Auto-capture 3x

&nbsp; Rotate 30-45° → Repeat 8-12 times

&nbsp; Press 's' → Save



SHORTCUTS:

&nbsp; SPACE   Auto-capture    v  Cartoon mode

&nbsp; 2       Curve mode      s  Save

&nbsp; ROI     Click \& drag    q  Quit



TARGET: 40,000-60,000 points for detailed model

═══════════════════════════════════════════════════════════

```



---



\*\*Ready to scan? Launch the scanner and start capturing!\*\* 🎯✨

