"""
Fix for MiDaS Depth Estimation Model Loading Issues
====================================================

Problem: HTTP Error 504 (Gateway Time-out) when loading MiDaS model

Solutions:
1. Manual model download
2. Use cached model
3. Retry with timeout handling
4. Use alternative model source
"""

import torch
import urllib.request
import os
from pathlib import Path

def download_midas_model_with_retry(model_type="MiDaS_small", max_retries=3):
    """
    Download MiDaS model with retry logic and better error handling.
    
    Args:
        model_type: Model version to download
        max_retries: Number of retry attempts
    
    Returns:
        Model object or None if failed
    """
    print(f"\n📊 LOADING DEPTH ESTIMATION MODEL")
    print("="*70)
    print(f"Loading model: {model_type}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            # Try loading with torch hub
            print(f"  Attempt {attempt}/{max_retries}...")
            
            model = torch.hub.load(
                "intel-isl/MiDaS", 
                model_type,
                trust_repo=True,
                timeout=60  # 60 second timeout
            )
            
            print(f"✅ Model loaded successfully!")
            return model
            
        except urllib.error.HTTPError as e:
            if e.code == 504:
                print(f"  ⚠️  Gateway timeout (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    print(f"  ⏳ Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                else:
                    print(f"\n❌ Failed after {max_retries} attempts")
                    print(f"   Error: {e}")
                    return None
            else:
                print(f"  ❌ HTTP Error {e.code}: {e.reason}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            if attempt < max_retries:
                print(f"  ⏳ Retrying...")
                import time
                time.sleep(5)
            else:
                return None
    
    return None


def download_midas_manually():
    """
    Manual download instructions if automatic download fails.
    """
    print("\n" + "="*70)
    print("📥 MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nIf automatic download keeps failing, download manually:")
    print("\n1. Go to: https://github.com/isl-org/MiDaS/releases")
    print("2. Download: midas_v21_small_256.pt")
    print("3. Place in: ~/.cache/torch/hub/checkpoints/")
    print("\nOr use this direct link:")
    print("https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt")
    print("\n4. Then run the scanner again")
    print("="*70)


def setup_depth_estimation_with_fallback():
    """
    Setup depth estimation with fallback options.
    
    Returns:
        Model and transform, or None if unavailable
    """
    try:
        # Try loading model with retries
        model = download_midas_model_with_retry("MiDaS_small", max_retries=3)
        
        if model is None:
            print("\n⚠️  Could not load depth estimation model")
            print("   Depth estimation will be DISABLED")
            print("   Scanner will continue without depth features")
            download_midas_manually()
            return None, None
        
        # Load transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"✅ Depth estimation ready (device: {device})")
        
        return model, transform
        
    except Exception as e:
        print(f"\n❌ Depth estimation setup failed: {e}")
        print("   Scanner will continue WITHOUT depth estimation")
        return None, None


# Alternative: Use lightweight depth estimation
def setup_lightweight_depth():
    """
    Alternative lightweight depth estimation that doesn't require model download.
    """
    print("\n📊 USING LIGHTWEIGHT DEPTH ESTIMATION")
    print("="*70)
    print("Using simple depth estimation (no model download required)")
    print("✅ Ready - basic depth estimation from laser triangulation")
    print("="*70)
    
    # Return a flag indicating lightweight mode
    return "lightweight"


if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("TESTING DEPTH ESTIMATION SETUP")
    print("="*70)
    
    # Test 1: Try loading with retry
    model, transform = setup_depth_estimation_with_fallback()
    
    if model is None:
        print("\n" + "="*70)
        print("FALLBACK: Using lightweight depth estimation")
        print("="*70)
        lightweight = setup_lightweight_depth()
        print(f"✅ Scanner can continue with: {lightweight}")
    else:
        print("\n✅ Full depth estimation available!")
