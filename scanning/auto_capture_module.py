#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-Capture Module for 3D Scanner
Captures 3 snapshots when quality metrics are optimal
Syncs with camera parameters panel (FPS, lighting, detection quality)
"""

import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime


class CaptureQualityMonitor:
    """
    Monitors capture quality metrics (FPS, lighting, detection quality).
    """
    
    def __init__(self, target_fps=15, min_brightness=100, max_brightness=200):
        """
        Initialize quality monitor.
        
        Args:
            target_fps: Target FPS for stable capture (default: 15)
            min_brightness: Minimum acceptable average brightness
            max_brightness: Maximum acceptable average brightness
        """
        self.target_fps = target_fps
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        
        # FPS tracking
        self.fps_history = []
        self.fps_window = 10  # Track last 10 frames
        
        # Frame timing
        self.last_frame_time = time.time()
        
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
            
            # Keep only recent history
            if len(self.fps_history) > self.fps_window:
                self.fps_history.pop(0)
        
        self.last_frame_time = current_time
        
        return self.get_current_fps()
    
    def get_current_fps(self):
        """Get average FPS from recent frames."""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)
    
    def check_brightness(self, frame):
        """
        Check if frame brightness is acceptable.
        
        Args:
            frame: BGR frame
            
        Returns:
            tuple: (is_good, brightness_value, status_text)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        is_good = self.min_brightness <= avg_brightness <= self.max_brightness
        
        if avg_brightness < self.min_brightness:
            status = "TOO DARK"
        elif avg_brightness > self.max_brightness:
            status = "TOO BRIGHT"
        else:
            status = "GOOD"
        
        return is_good, avg_brightness, status
    
    def check_fps_stable(self):
        """
        Check if FPS is stable and near target.
        
        Returns:
            tuple: (is_stable, current_fps, status_text)
        """
        if len(self.fps_history) < self.fps_window:
            return False, 0, "WARMING UP"
        
        current_fps = self.get_current_fps()
        fps_variance = np.std(self.fps_history)
        
        # Stable if within 20% of target and low variance
        is_stable = (
            abs(current_fps - self.target_fps) < self.target_fps * 0.2 and
            fps_variance < 5
        )
        
        status = "STABLE" if is_stable else "UNSTABLE"
        return is_stable, current_fps, status
    
    def check_laser_quality(self, frame, laser_detector_func=None):
        """
        Check laser detection quality.
        
        Args:
            frame: BGR frame
            laser_detector_func: Optional function that returns (detected, confidence)
            
        Returns:
            tuple: (is_good, confidence, status_text)
        """
        if laser_detector_func:
            detected, confidence = laser_detector_func(frame)
            is_good = detected and confidence > 0.7
            status = "DETECTED" if detected else "NO LASER"
            return is_good, confidence, status
        
        # Fallback: Check for red dot presence
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_pixels = np.sum(red_mask > 0)
        
        is_good = red_pixels > 50  # At least 50 red pixels
        confidence = min(red_pixels / 500.0, 1.0)  # Normalize to 0-1
        status = "DETECTED" if is_good else "NO LASER"
        
        return is_good, confidence, status
    
    def check_all_conditions(self, frame, laser_detector_func=None):
        """
        Check all capture conditions.
        
        Returns:
            tuple: (all_good, status_dict)
        """
        fps_good, current_fps, fps_status = self.check_fps_stable()
        bright_good, brightness, bright_status = self.check_brightness(frame)
        laser_good, laser_conf, laser_status = self.check_laser_quality(frame, laser_detector_func)
        
        all_good = fps_good and bright_good and laser_good
        
        status = {
            'fps': {'good': fps_good, 'value': current_fps, 'status': fps_status},
            'brightness': {'good': bright_good, 'value': brightness, 'status': bright_status},
            'laser': {'good': laser_good, 'value': laser_conf, 'status': laser_status},
            'ready': all_good
        }
        
        return all_good, status


class AutoCaptureModule:
    """
    Handles automatic 3-point capture with quality-based timing.
    """
    
    def __init__(self, window_name="3D Scanner", capture_count=3, 
                 quality_check_enabled=True, max_wait_seconds=10.0):
        """
        Initialize auto-capture module.
        
        Args:
            window_name: OpenCV window name for display
            capture_count: Number of captures to perform (default: 3)
            quality_check_enabled: Wait for good quality before capturing
            max_wait_seconds: Maximum wait time for quality conditions
        """
        self.window_name = window_name
        self.capture_count = capture_count
        self.quality_check_enabled = quality_check_enabled
        self.max_wait_seconds = max_wait_seconds
        
        self.captured_frames = []
        self.captured_metadata = []
        
        # Initialize quality monitor
        self.quality_monitor = CaptureQualityMonitor()
        
    def draw_status_panel(self, frame, status_dict):
        """
        Draw quality status panel (synced with bottom-right corner panel).
        
        Args:
            frame: Video frame
            status_dict: Dictionary with quality status from monitor
            
        Returns:
            Frame with status panel drawn
        """
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Panel dimensions (bottom-right corner)
        panel_w, panel_h = 280, 120
        panel_x = w - panel_w - 10
        panel_y = h - panel_h - 10
        
        # Semi-transparent background
        overlay = overlay_frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w].copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, 
                       overlay_frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w], 
                       0.3, 0, overlay_frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w])
        
        # Border
        cv2.rectangle(overlay_frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (0, 255, 255), 2)
        
        # Title
        cv2.putText(overlay_frame, "CAPTURE QUALITY", 
                   (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Status lines
        y_offset = 50
        line_height = 22
        
        for key, label in [('fps', 'FPS'), ('brightness', 'Light'), ('laser', 'Laser')]:
            info = status_dict.get(key, {})
            is_good = info.get('good', False)
            value = info.get('value', 0)
            status = info.get('status', 'N/A')
            
            # Status indicator (green checkmark or red X)
            indicator = "✓" if is_good else "✗"
            color = (0, 255, 0) if is_good else (0, 0, 255)
            
            # Draw line
            if key == 'fps':
                text = f"{indicator} {label}: {value:.1f} - {status}"
            elif key == 'brightness':
                text = f"{indicator} {label}: {value:.0f} - {status}"
            else:  # laser
                text = f"{indicator} {label}: {status}"
            
            cv2.putText(overlay_frame, text, 
                       (panel_x + 10, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height
        
        # Overall status
        ready = status_dict.get('ready', False)
        ready_text = "READY TO CAPTURE" if ready else "WAITING..."
        ready_color = (0, 255, 0) if ready else (0, 165, 255)
        
        cv2.putText(overlay_frame, ready_text, 
                   (panel_x + 10, panel_y + panel_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ready_color, 2)
        
        return overlay_frame
    
    def show_progress_overlay(self, frame, message, progress, total, status_dict=None):
        """
        Draw progress bar and status message on frame.
        """
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw status panel first (if provided)
        if status_dict:
            overlay_frame = self.draw_status_panel(overlay_frame, status_dict)
        
        # Create semi-transparent dark overlay for message box
        overlay = np.zeros_like(overlay_frame)
        cv2.addWeighted(overlay, 0.6, overlay_frame, 0.4, 0, overlay_frame)
        
        # Draw main message box
        box_w, box_h = 600, 200
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        
        # Box background
        cv2.rectangle(overlay_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                     (40, 40, 40), -1)
        cv2.rectangle(overlay_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                     (0, 255, 255), 3)
        
        # Message text
        cv2.putText(overlay_frame, message, (box_x + 50, box_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Progress text
        progress_text = f"Capture {progress}/{total}"
        cv2.putText(overlay_frame, progress_text, (box_x + 180, box_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Progress bar
        bar_w, bar_h = 500, 30
        bar_x = box_x + 50
        bar_y = box_y + 140
        
        # Background bar
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     (80, 80, 80), -1)
        
        # Progress fill
        fill_w = int(bar_w * (progress / total))
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                     (0, 255, 0), -1)
        
        # Percentage text
        percentage = int((progress / total) * 100)
        cv2.putText(overlay_frame, f"{percentage}%", (bar_x + bar_w//2 - 30, bar_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay_frame
    
    def wait_for_quality(self, cap, capture_num, total, laser_detector_func=None):
        """
        Wait for capture quality conditions to be met.
        
        Args:
            cap: VideoCapture object
            capture_num: Current capture number
            total: Total captures
            laser_detector_func: Optional laser detection function
            
        Returns:
            tuple: (frame, status_dict, success)
        """
        print(f"\n    ⏳ Waiting for optimal capture conditions...")
        
        start_time = time.time()
        
        while (time.time() - start_time) < self.max_wait_seconds:
            ret, frame = cap.read()
            if not ret:
                return None, None, False
            
            # Update FPS
            self.quality_monitor.update_fps()
            
            # Check all conditions
            all_good, status_dict = self.quality_monitor.check_all_conditions(
                frame, laser_detector_func
            )
            
            # Draw status panel
            display_frame = self.draw_status_panel(frame, status_dict)
            
            # Add waiting message
            h, w = display_frame.shape[:2]
            wait_text = f"Waiting for quality... ({capture_num}/{total})"
            cv2.putText(display_frame, wait_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            # Check if conditions met
            if all_good:
                print(f"    ✅ Quality conditions met!")
                print(f"       FPS: {status_dict['fps']['value']:.1f} - {status_dict['fps']['status']}")
                print(f"       Light: {status_dict['brightness']['value']:.0f} - {status_dict['brightness']['status']}")
                print(f"       Laser: {status_dict['laser']['status']}")
                return frame, status_dict, True
            
            cv2.waitKey(30)  # Check at ~30 FPS
        
        # Timeout
        print(f"    ⚠️  Timeout waiting for quality (captured anyway)")
        ret, frame = cap.read()
        _, status_dict = self.quality_monitor.check_all_conditions(frame, laser_detector_func)
        return frame, status_dict, True
    
    def capture_sequence(self, cap, process_frame_callback=None, laser_detector_func=None):
        """
        Capture sequence of frames with quality-based timing.
        
        Args:
            cap: OpenCV VideoCapture object
            process_frame_callback: Optional callback function(frame, capture_num) 
            laser_detector_func: Optional function(frame) -> (detected, confidence)
        
        Returns:
            tuple: (captured_frames, metadata_list, success)
        """
        self.captured_frames = []
        self.captured_metadata = []
        
        print(f"\n{'='*80}")
        print(f"🎬 AUTO-CAPTURE STARTED: {self.capture_count} quality-based captures")
        print(f"{'='*80}")
        
        for capture_num in range(1, self.capture_count + 1):
            print(f"\n📸 Capture {capture_num}/{self.capture_count}...")
            
            # Wait for quality conditions if enabled
            if self.quality_check_enabled:
                frame, status_dict, success = self.wait_for_quality(
                    cap, capture_num, self.capture_count, laser_detector_func
                )
                
                if not success or frame is None:
                    print(f"    ❌ Failed to capture frame {capture_num}")
                    return self.captured_frames, self.captured_metadata, False
            else:
                # Immediate capture (legacy behavior)
                ret, frame = cap.read()
                if not ret:
                    return self.captured_frames, self.captured_metadata, False
                status_dict = {}
            
            # Show "CAPTURING" message
            message = f"CAPTURING {capture_num}/{self.capture_count}..."
            display_frame = self.show_progress_overlay(
                frame, message, capture_num, self.capture_count, status_dict
            )
            cv2.imshow(self.window_name, display_frame)
            cv2.waitKey(500)  # Show for 0.5 seconds
            
            # Store captured frame
            self.captured_frames.append(frame.copy())
            
            # Create metadata
            metadata = {
                'capture_num': capture_num,
                'timestamp': datetime.now().isoformat(),
                'frame_shape': frame.shape,
                'capture_time': time.time(),
                'quality_status': status_dict
            }
            self.captured_metadata.append(metadata)
            
            # Process frame if callback provided
            if process_frame_callback:
                try:
                    process_frame_callback(frame, capture_num)
                    print(f"    ✅ Processed capture {capture_num}")
                except Exception as e:
                    print(f"    ⚠️  Processing error: {e}")
            else:
                print(f"    ✅ Captured frame {capture_num}")
        
        # Show completion message
        ret, frame = cap.read()
        if ret:
            message = "AUTO-CAPTURE COMPLETE!"
            display_frame = self.show_progress_overlay(
                frame, message, self.capture_count, self.capture_count
            )
            cv2.imshow(self.window_name, display_frame)
            cv2.waitKey(1000)  # Show for 1 second
        
        print(f"\n{'='*80}")
        print(f"✅ AUTO-CAPTURE COMPLETE: {len(self.captured_frames)} frames captured")
        print(f"{'='*80}\n")
        
        return self.captured_frames, self.captured_metadata, True
    
    def save_captures(self, output_dir, prefix="capture"):
        """
        Save captured frames to disk.
        
        Args:
            output_dir: Directory to save frames
            prefix: Filename prefix (default: "capture")
        
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (frame, metadata) in enumerate(zip(self.captured_frames, self.captured_metadata)):
            filename = f"{prefix}_{timestamp}_{i+1:03d}.png"
            filepath = output_path / filename
            
            cv2.imwrite(str(filepath), frame)
            saved_files.append(filepath)
            print(f"    💾 Saved: {filepath}")
        
        return saved_files
    
    def get_frames(self):
        """Get list of captured frames."""
        return self.captured_frames
    
    def get_metadata(self):
        """Get list of capture metadata."""
        return self.captured_metadata
    
    def clear(self):
        """Clear captured frames and metadata."""
        self.captured_frames = []
        self.captured_metadata = []


# Example usage and testing
def test_auto_capture():
    """Test the auto-capture module."""
    print("\n" + "="*80)
    print("🧪 TESTING AUTO-CAPTURE MODULE")
    print("="*80)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Initialize auto-capture
    auto_capture = AutoCaptureModule(
        window_name="Auto-Capture Test",
        capture_count=3,
        interval_seconds=1.0
    )
    
    print("\nPress SPACE to start auto-capture sequence...")
    print("Press Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Auto-Capture Test", frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord(' '):  # Space bar
            frames, metadata, success = auto_capture.capture_sequence(cap)
            
            if success:
                print(f"\n✅ Captured {len(frames)} frames")
                
                # Save frames
                output_dir = Path(__file__).parent / "test_captures"
                saved_files = auto_capture.save_captures(output_dir)
                print(f"\n💾 Saved {len(saved_files)} files to: {output_dir}")
                
                # Clear for next capture
                auto_capture.clear()
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_auto_capture()