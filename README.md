<img width="2185" height="1605" alt="snapshot_00-001" src="https://github.com/user-attachments/assets/14f8ff57-bea6-4fd5-b153-a9afef143443" />
<img width="2071" height="1605" alt="snapshot001" src="https://github.com/user-attachments/assets/cb120ab0-c3b9-4036-a9c6-a5a5167571db" />
# Point Cloud Detection Scanner 🔴🎯

**Advanced 3D Scanner optimized for full spectrum dot distance detection (635nm Red Laser)**

Laser triangulation-based 3D scanner with AI depth estimation, real-time visualization, and mesh generation.

---

## 🚀 Features

### **Core Scanning Modes**
- **🔴 Mode 1: Red Laser (635nm)** - Precise laser dot triangulation for high-accuracy 3D scanning
- **🌈 Mode 2: Curve Trace** - Detect and trace continuous curves/contours
- **📐 Mode 3: Corner Detection** - Find corners and feature points
- **🤖 Mode 4: AI Depth** - Monocular depth estimation using MiDaS neural network

### **3D Visualization**
- **Interactive 3D Viewer** (Press `O`) - Open3D-powered point cloud visualization
  - Rotate, pan, zoom controls
  - Color-coded by height
  - Coordinate frame reference
  - Positioned in top-right corner to avoid window conflicts

### **Mesh Generation**
- **Poisson Surface Reconstruction** - Watertight, smooth meshes
- **Ball Pivoting Algorithm (BPA)** - Faithful to original data
- **Screened Algorithm**
- Auto-generates mesh on save (`.obj` + `.ply` formats)

### **Advanced Features**
- **Lazy Loading** - 60-80% faster startup (loads AI modules only when needed)
- **Spectrum Analyzer** - Multi-wavelength laser detection (380-1000nm)
- **Camera Calibration** - Automatic distortion correction
- **Auto-capture Mode** - 3-snapshot rotation workflow
- **ROI (Region of Interest)** - Crop scan area with scissors tool
- **GPU Acceleration** - CUDA support for undistortion & depth estimation
- **Quality Monitoring** - Real-time sharpness & brightness analysis


## 📦 Installation

### **Requirements**
- Python 3.8+
- Webcam (1280x720 recommended)

### **Quick Install** (Basic laser scanning):
```bash
pip install opencv-python numpy
```

### **Full Install** (AI depth + 3D viewer):
```bash
pip install -r requirements.txt
```

### **GPU Acceleration** (Optional - 10x faster AI depth):
```bash
# Install CUDA 11.8 or 12.x from NVIDIA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy timm open3d psutil
```

---

## 🎮 Usage

### **Launch Scanner**
```bash
cd scanning
python laser_3d_scanner_advanced.py
```

### **Keyboard Controls**

| Key | Action | Description |
|-----|--------|-------------|
| **1/2/3/4** | Mode Switch | Full Spectrum / Curve / Corners / AI Depth |
| **SPACE** | Capture | Add points to 3D cloud |
| **O** | 3D Viewer | Open interactive point cloud visualization |
| **S** | Save | Export `.ply` + auto-generate mesh |
| **C** | Clear | Delete all captured points |
| **M** | Mesh Method | Toggle Poisson / Ball Pivoting |
| **V** | Cartoon Mode | Toggle camera cartoon-style settings |
| **P** | Spectrum Cycle | Switch laser wavelength (635nm / 532nm / 450nm / IR / Full) |
| **+/-** | Curve Sample | Adjust curve point sampling rate |
| **[/]** | Corner Count | Adjust max corner detection limit |
| **,/.** | Edge Threshold | Adjust Canny edge sensitivity |
| **I** | AI Panel | Toggle AI quality panel visibility |
| **B** | Controls Panel | Toggle keyboard controls display |
| **Q/ESC** | Quit | Exit scanner |

### **Depth Mode** (Mode 4 - AI):
| Key | Action |
|-----|--------|
| **Z** | Toggle depth visualization overlay |
| **X** | Toggle sparse/dense point cloud |
| **W/E** | Adjust min depth range |
| **R/F** | Adjust max depth range |

---

## 📁 Project Structure

```
point_cloud_detection/
├── scanning/                    # Main scanner module
│   ├── laser_3d_scanner_advanced.py  # Main scanner (2600+ lines)
│   ├── depth_estimator.py       # AI depth estimation (MiDaS)
│   ├── spectrum_config.py       # Multi-wavelength laser detection
│   ├── panel_display_module.py  # UI panels and overlays
│   ├── gpu_optimizer.py         # CUDA acceleration
│   ├── calibration_helper.py    # Auto-setup for new users
│   ├── camera_identifier.py     # Camera fingerprinting
│   ├── SCANNER_MATRIX.py        # Fast lookup reference
│   └── data/                    # Scan output folder
│       └── point_clouds/        # .ply, .obj, .npz files
│
├── calibration/                 # Camera calibration tools
│   ├── camera_distance_detector_calibrated.py
│   └── checkerboard.py          # Calibration pattern generator
│
├── ai_analysis/                 # AI quality analysis
│   ├── camera_info.py           # FPS, exposure, resolution
│   ├── image_quality.py         # Sharpness, brightness detection
│   └── optimized_analyzer.py    # GPU-accelerated analysis
│
├── utils/                       # Utilities
│   ├── project_manager.py       # Scan project organization
│   └── system_requirements.py   # Dependency checker
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
└── README.md                    # This file
```

---

## 🎯 Workflow Example

### **1. First-Time Setup**
```bash
# Calibrate camera (auto-generates checkerboard)
python calibration/checkerboard.py
# Follow on-screen instructions for 15-20 images

# Alternative: Use default calibration (less accurate)
# Scanner auto-detects and offers to generate checkerboard
```

### **2. Basic Scanning**
1. Launch scanner: `python laser_3d_scanner_advanced.py
2. Press **SPACE** to capture points
3. Rotate object, capture more points  
4. Press **S** to save (auto-generates mesh)

### **3. AI Depth Scanning** (No laser required)
1. Press **4** for AI Depth mode
2. Position camera to view object
3. Press **SPACE** to capture dense depth map
4. Press **S** to save

---

## 🔧 Configuration

### **Camera Settings**
Edit `laser_3d_scanner_advanced.py`:
```python
# Line ~1711
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### **Laser Color** (for non-635nm lasers)
Edit `spectrum_config.py` or press **P** to cycle presets:
- Full Spectrum - Default
- 532nm Green
- 450nm Blue
- 780nm Near-IR
- Full Spectrum (380-1000nm)

### **Save Location**
On first run, scanner prompts for save folder:
- Option 1: `scanning/` folder
- Option 2: Custom path (enter manually)

---

## 📊 Output Files

### **Point Clouds**
- `scan_3d_YYYYMMDD_HHMMSS.ply` - Colored point cloud (ASCII)
- `scan_3d_bosch_glm42.npz` - Compressed NumPy format

### **Meshes** (Auto-generated)
- `scan_*_mesh.obj` - Poisson or BPA mesh
- `scan_*_mesh.ply` - Mesh in PLY format

### **Metadata**
- Rotation angles
- Session timestamps
- Calibration fingerprint

---

## 🐛 Troubleshooting

### **"Depth estimation unavailable"**
```bash
# Install PyTorch + dependencies
pip install torch torchvision timm
```

### **"Camera not found"**
```python
# Change camera index in laser_3d_scanner_advanced.py
WEBCAM_INDEX = 1  # Try 0, 1, 2...
```

### **"Calibration file not found"**
Scanner auto-generates checkerboard and guides setup. Or:
```bash
python calibration/checkerboard.py
```

### **Slow AI depth mode**
Enable GPU acceleration (see Installation → GPU Acceleration)

### **3D Viewer shrinks video window**
Fixed in v2.0 - viewer now positions in top-right corner

---

## 📝 Technical Details

### **Laser Triangulation**
- **Method**: Dot centroid detection with sub-pixel accuracy
- **Range**: Configurable via calibration (typically 20-200cm)
- **Accuracy**: ±2mm at 1m distance (with calibration)

### **AI Depth Estimation**
- **Model**: MiDaS DPT-Large (Intel ISL)
- **Input**: Single RGB image
- **Output**: Dense depth map (downsampled 2x-4x)
- **Speed**: ~2 FPS (CPU), ~15 FPS (GPU RTX 3060)

### **Mesh Algorithms**
- **Poisson**: 8-10 octree depth, watertight surfaces
- **BPA**: 5mm ball radius, preserves fine details

---

## 🔗 Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| opencv-python | Camera capture, image processing | ✅ Yes |
| numpy | Array operations, point clouds | ✅ Yes |
| torch | AI depth neural network | ⚠️ Optional |
| torchvision | Image transforms for AI | ⚠️ Optional |
| timm | MiDaS DPT model support | ⚠️ Optional |
| open3d | 3D visualization, mesh generation | ⚠️ Optional |
| psutil | System monitoring | ⚠️ Optional |

---

## 📜 License

See [LICENSE](LICENSE) file.

---

## 🤝 Contributing

This is a private repository for development. Contributions welcome after review.

---

## 📧 Support

For issues or questions, check:
1. Troubleshooting section above
2. Code comments in `laser_3d_scanner_advanced.py`
3. `SCANNER_MATRIX.py` for quick reference

---

**Last Updated**: December 13, 2025  
**Version**: 2.0 (Lazy Loading + 3D Viewer Update)

