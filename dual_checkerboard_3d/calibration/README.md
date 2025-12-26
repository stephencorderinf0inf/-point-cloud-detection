# Camera Calibration Files

This folder contains camera calibration data for accurate 3D scanning.

## Files

- **`camera_calibration_default.npz`** - Generic calibration for standard webcams (1280x720)
  - Focal length: 800px (typical for most webcams)
  - Principal point: Center of image (640, 360)
  - Minimal distortion coefficients
  - **Use this for quick testing or if you don't have a calibration**

- **`camera_calibration.npz`** - Your custom calibration (create with checkerboard.py)
  - More accurate for your specific camera
  - Recommended for production scans

## When to Use Which?

### Use Default Calibration When:
✅ Quick testing/demo  
✅ Don't have printer for checkerboard  
✅ Camera roughly matches standard webcam specs  
✅ Accuracy isn't critical  

### Use Custom Calibration When:
✅ Production/research work  
✅ Need high accuracy  
✅ Non-standard camera (fisheye, phone, etc.)  
✅ Camera has significant distortion  

## Create Custom Calibration

Run the calibration wizard:

```bash
cd dual_checkerboard_3d
python checkerboard.py
```

Follow these steps:
1. Print the generated checkerboard pattern
2. Capture 15-20 images at different angles
3. Calibration data saves automatically

**Time required:** 15-30 minutes  
**Accuracy improvement:** 20-40% better than default

## Technical Details

### Default Calibration Parameters
```python
Camera Matrix:
[[800   0  640]
 [  0 800  360]
 [  0   0    1]]

Distortion Coefficients:
[0.1, -0.05, 0.0, 0.0, 0.0]
```

- **Focal Length (fx, fy):** 800px (assumes ~60° FOV)
- **Principal Point (cx, cy):** Image center
- **Distortion:** Minimal barrel distortion

### Custom Calibration
Your custom calibration accounts for:
- Actual focal length of your camera
- Lens distortion (barrel/pincushion)
- Sensor position variations
- Manufacturing tolerances

## Troubleshooting

**Scanner says "No calibration found"**
- Install.bat should have created `camera_calibration_default.npz`
- If missing, run: `python dual_checkerboard_3d/checkerboard.py`

**Poor scan accuracy**
- Default calibration may not match your camera
- Create custom calibration for better results

**Multiple calibration files**
- Scanner auto-detects and lets you choose
- Rename to `camera_calibration_<camera_name>.npz` for auto-selection
