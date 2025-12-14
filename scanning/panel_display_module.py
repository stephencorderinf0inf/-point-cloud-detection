#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel Display Module for 3D Scanner
Handles all on-screen information panels (top-left controls + bottom-right AI/quality)
"""

import cv2
import numpy as np
import time


class PanelDisplayModule:
    """
    Manages all overlay panels for the scanner interface.
    """
    
    def __init__(self, window_name="3D Scanner"):
        """
        Initialize panel display module.
        
        Args:
            window_name: OpenCV window name for display
        """
        self.window_name = window_name
        self.info_box_visible = True
        self.ai_panel_visible = True
        self.terminal_visible = True  # Terminal output panel visibility
        
        # FPS tracking
        self.fps_history = []
        self.fps_window = 30
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        
        # Message bar tracking
        self.message_queue = []
        self.max_messages = 10  # Show more messages for scrolling
        self.message_timeout = None  # No timeout - messages persist
        self.scroll_offset = 0  # Scroll position (0 = newest messages)
    
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
            
            if len(self.fps_history) > self.fps_window:
                self.fps_history.pop(0)
            
            if self.fps_history:
                self.current_fps = sum(self.fps_history) / len(self.fps_history)
        
        self.last_frame_time = current_time
        return self.current_fps
    
    def add_message(self, text, color=(255, 255, 0)):
        """
        Add a message to the message bar queue.
        
        Args:
            text: Message text to display
            color: BGR color tuple (default: cyan)
        """
        timestamp = time.time()
        self.message_queue.append({
            'text': text,
            'color': color,
            'timestamp': timestamp
        })
        
        # Keep scrolling window of recent messages
        if len(self.message_queue) > 50:  # Keep last 50 for scrollback
            self.message_queue.pop(0)
    
    def clear_old_messages(self):
        """Remove messages older than timeout (if timeout is set)."""
        if self.message_timeout is None:
            return  # No timeout - keep all messages
        
        current_time = time.time()
        self.message_queue = [msg for msg in self.message_queue 
                             if current_time - msg['timestamp'] < self.message_timeout]
    
    def scroll_terminal_up(self):
        """Scroll terminal output up (show older messages)."""
        max_scroll = max(0, len(self.message_queue) - self.max_messages)
        self.scroll_offset = min(self.scroll_offset + 1, max_scroll)
        return self.scroll_offset
    
    def scroll_terminal_down(self):
        """Scroll terminal output down (show newer messages)."""
        self.scroll_offset = max(0, self.scroll_offset - 1)
        return self.scroll_offset
    
    def scroll_to_bottom(self):
        """Reset scroll to show newest messages."""
        self.scroll_offset = 0
    
    def draw_message_bar(self, frame):
        """
        Draw scrollable terminal log at the bottom of screen.
        Positioned below both side panels (controls and camera info).
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with message bar drawn
        """
        # Return early if terminal is hidden
        if not self.terminal_visible:
            return frame
        
        # Clean old messages (if timeout enabled)
        self.clear_old_messages()
        
        if not self.message_queue:
            return frame
        
        h, w = frame.shape[:2]
        
        # Terminal log at absolute bottom - spanning full width
        bar_height = 200  # ~10 lines at 18px each + margins
        bar_y = h - bar_height  # Bottom of screen
        bar_x_left = 360  # Start after control panel
        bar_x_right = w - 300  # Stop before camera info panel (280px + margin)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x_left, bar_y), (bar_x_right, h - 10), 
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        
        # Border
        cv2.rectangle(frame, (bar_x_left, bar_y), (bar_x_right, h - 10), 
                     (255, 255, 0), 2)  # Cyan border
        
        # Title
        cv2.putText(frame, "TERMINAL OUTPUT", 
                   (bar_x_left + 10, bar_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Scroll indicator in title bar
        if self.scroll_offset > 0:
            scroll_text = f"[Scrolled up {self.scroll_offset}]"
            cv2.putText(frame, scroll_text,
                       (bar_x_left + 180, bar_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw messages with scroll offset
        line_height = 18
        y_offset = bar_y + 45
        
        # Calculate visible range based on scroll offset
        total_messages = len(self.message_queue)
        if total_messages > 0:
            # When scrolled, show older messages
            end_idx = total_messages - self.scroll_offset
            start_idx = max(0, end_idx - self.max_messages)
            visible_messages = self.message_queue[start_idx:end_idx]
        else:
            visible_messages = []
        
        for msg in visible_messages:
            # Truncate long messages to fit width
            text = msg['text']
            max_chars = int((bar_x_right - bar_x_left - 20) / 7)  # Approximate char width
            if len(text) > max_chars:
                text = text[:max_chars-3] + "..."
            
            cv2.putText(frame, text, 
                       (bar_x_left + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, msg['color'], 1)
            y_offset += line_height
        
        # Show scroll controls and indicator
        controls_text = "[[ / ]]: Scroll  [\\]: Bottom"
        cv2.putText(frame, controls_text,
                   (bar_x_right - 280, bar_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Show count if scrolled or if more messages exist above
        if self.scroll_offset > 0 or len(self.message_queue) > self.max_messages:
            total_above = len(self.message_queue) - len(visible_messages) - self.scroll_offset
            if total_above > 0:
                count_text = f"({total_above} more above)"
                cv2.putText(frame, count_text,
                           (bar_x_left + 10, h - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        return frame
    
    def draw_scanner_controls_panel(self, frame, current_angle=0, rotation_step=30, 
                                   current_session=0, points_count=0, current_mode="Unknown",
                                   curves_info=None, sensitivity_info=None):  # 🎨 NEW PARAMETER
        """
        Draw scanner controls panel in top-left corner.
        
        Args:
            frame: Video frame
            current_angle: Current rotation angle
            rotation_step: Rotation step size
            current_session: Current capture session
            points_count: Total points captured
            current_mode: Current detection mode name
            curves_info: Dict with 'count' and 'points' for curve mode
            sensitivity_info: Dict with 'curve_rate', 'corner_max', 'canny_low', 'canny_high'
            
        Returns:
            Frame with panel drawn
        """
        if not self.info_box_visible:
            # Just show minimal hint
            hint_text = f"[B] Show Controls | Angle:{current_angle:.0f}deg Pts:{points_count}"
            cv2.putText(frame, hint_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return frame
        
        h, w = frame.shape[:2]
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 340, 700 # Reduced from 360x710
        
        # Ensure panel fits on screen
        if panel_y + panel_h > h:
            panel_h = h - panel_y - 10
        
        # Semi-transparent background
        panel_region = frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w].copy()
        cv2.rectangle(panel_region, (0, 0), (panel_w, panel_h), (40, 40, 40), -1)
        cv2.addWeighted(panel_region, 0.8, 
                       frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w], 
                       0.2, 0, frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w])
        
        # Border (CYAN)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (255, 255, 0), 2)  # Cyan in BGR
        
        # Title
        cv2.putText(frame, "SCANNER CONTROLS", 
                   (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        
        # Hide button hint
        cv2.putText(frame, "[B] Hide", 
                   (panel_x + panel_w - 80, panel_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        
        y_offset = 42
        line_height = 18  # Reduced from 20
        
        # === STATUS SECTION ===
        cv2.putText(frame, f"Angle: {current_angle:.1f}deg", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Session: {current_session}", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Step: {rotation_step:.1f}deg", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Points: {points_count}", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Mode: {current_mode}", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        y_offset += line_height + 4
        
        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + y_offset - 3), 
                (panel_x + panel_w - 10, panel_y + y_offset - 3), 
                (100, 100, 100), 1)
        y_offset += 8
        
        # === ROTATION CONTROLS ===
        cv2.putText(frame, "ROTATION:", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        y_offset += line_height
        
        rotation_controls = [
            "[SPACE] Capture point(s)",
            f"[R] Rotate +{rotation_step:.1f}deg",
            "[T] Set angle manually",
            "[E] Change step size"
        ]
        
        for line in rotation_controls:
            color = (0, 255, 0) if "SPACE" in line else (255, 255, 0)
            weight = 2 if "SPACE" in line else 1
            cv2.putText(frame, line, 
                       (panel_x + 20, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, weight)
            y_offset += line_height
        
        y_offset += 3
        
        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + y_offset - 3), 
                (panel_x + panel_w - 10, panel_y + y_offset - 3), 
                (100, 100, 100), 1)
        y_offset += 6
        
        # === MODE SWITCHING ===
        cv2.putText(frame, "MODES:", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1)
        y_offset += line_height
        
        mode_controls = [
            "[1] Laser detection",
            "[2] Curve tracing",
            "[3] Corner detection",
            "[4] AI Depth (MiDaS)"
        ]
        
        for line in mode_controls:
            cv2.putText(frame, line, 
                       (panel_x + 20, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1)
            y_offset += line_height
        
        y_offset += 3
        
        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + y_offset - 3), 
                (panel_x + panel_w - 10, panel_y + y_offset - 3), 
                (100, 100, 100), 1)
        y_offset += 6
        
        # === UTILITY CONTROLS ===
        cv2.putText(frame, "UTILITIES:", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        y_offset += line_height
        
        utility_controls = [
            "[V] Cartoon mode",
            "[D] Debug view",
            "[C] Clear points",
            "[O] 3D Viewer",
            "[F] Process photos",
            "[T] Toggle capture mode",
            "[S] Save cloud",
            "[M] Mesh method",
            "[I] Toggle AI panel",
            "[H] Hide terminal",
            "[[ / ]] Scroll terminal",
            "[Q] Quit"
        ]
        
        for line in utility_controls:
            cv2.putText(frame, line, 
                       (panel_x + 20, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            y_offset += line_height
        
        y_offset += 3
        
        # 🎨 NEW: SENSITIVITY SECTION
        cv2.line(frame, (panel_x + 10, panel_y + y_offset - 3), 
                (panel_x + panel_w - 10, panel_y + y_offset - 3), 
                (100, 100, 100), 1)
        y_offset += 6
        
        cv2.putText(frame, "SENSITIVITY:", 
                   (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 200, 255), 1)
        y_offset += line_height
        
        # Show sensitivity controls
        sensitivity_controls = [
            "[+/-] Curve sampling",
            "[{/}] Corner count",
            "[</>] Edge threshold"
        ]
        
        for line in sensitivity_controls:
            cv2.putText(frame, line, 
                       (panel_x + 20, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1)
            y_offset += line_height
        
        # Show CURRENT VALUES if sensitivity_info provided
        if sensitivity_info:
            y_offset += 3
            
            curve_rate = sensitivity_info.get('curve_rate', 5)
            corner_max = sensitivity_info.get('corner_max', 100)
            canny_low = sensitivity_info.get('canny_low', 50)
            canny_high = sensitivity_info.get('canny_high', 150)
            
            cv2.putText(frame, f"Curve: 1/{curve_rate}", 
                       (panel_x + 30, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)
            y_offset += 15
            
            cv2.putText(frame, f"Corners: {corner_max}", 
                       (panel_x + 30, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)
            y_offset += 15
            
            cv2.putText(frame, f"Edges: {canny_low}/{canny_high}", 
                       (panel_x + 30, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)
            y_offset += 15
        
        y_offset += 4
        
        # === CURVE MODE INFO (if provided) ===
        if curves_info and current_mode == "Curve":
            cv2.line(frame, (panel_x + 10, panel_y + y_offset - 3), 
                    (panel_x + panel_w - 10, panel_y + y_offset - 3), 
                    (100, 100, 100), 1)
            y_offset += 8
            
            cv2.putText(frame, "CURVE MODE:", 
                       (panel_x + 15, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            y_offset += line_height
            
            curve_count = curves_info.get('count', 0)
            curve_points = curves_info.get('points', 0)
            
            if curve_count > 0:
                cv2.putText(frame, f"Curves: {curve_count}", 
                           (panel_x + 20, panel_y + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
                y_offset += line_height
                cv2.putText(frame, f"Curve pts: {curve_points}", 
                           (panel_x + 20, panel_y + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "No curves detected", 
                           (panel_x + 20, panel_y + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        
        return frame
    
    def draw_ai_quality_panel(self, frame, ai_result=None):
        """
        Draw AI/quality panel in bottom-right corner.
        
        Args:
            frame: Video frame
            ai_result: Dictionary with AI analysis results (optional)
            
        Returns:
            Frame with panel drawn
        """
        # FIXED: Return BEFORE any drawing if panel is hidden
        if not self.ai_panel_visible:
            return frame
        
        h, w = frame.shape[:2]
        
        # Update FPS
        self.update_fps()
        
        # Panel dimensions
        panel_w, panel_h = 280, 200
        panel_x = w - panel_w - 10
        panel_y = h - panel_h - 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (255, 255, 0), 2)  # Cyan border
        
        # Title
        cv2.putText(frame, "Camera Info", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # FPS (always show)
        fps_color = (0, 255, 0) if self.current_fps > 15 else (0, 255, 255) if self.current_fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (panel_x + 10, panel_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Show AI results if available
        if ai_result:
            y_offset = 70
            line_height = 20
            
            cv2.putText(frame, f"Res: {ai_result.get('resolution', 'N/A')}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Focal: {ai_result.get('focal_length', 'N/A')}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += line_height
            
            sharpness = ai_result.get('sharpness', 0)
            cv2.putText(frame, f"Sharp: {sharpness:.2f}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if sharpness > 0.5 else (255, 128, 0), 1)
            y_offset += line_height
            
            brightness = ai_result.get('brightness', 0)
            cv2.putText(frame, f"Bright: {brightness:.2f}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if brightness > 0.4 else (255, 128, 0), 1)
            y_offset += line_height
            
            status = ai_result.get('status', 'Unknown')
            cv2.putText(frame, f"Status: {status}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if status == "Excellent" else (255, 255, 0), 1)
        else:
            cv2.putText(frame, "Analyzing...", (panel_x + 10, panel_y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Hide hint
        cv2.putText(frame, "Press 'i' to hide", (panel_x + 10, panel_y + panel_h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def draw_all_panels(self, frame, scanner_params, ai_result=None):
        """
        Draw all panels on frame (convenience method).
        
        Args:
            frame: Video frame
            scanner_params: Dict with scanner state including sensitivity_info
            ai_result: AI analysis results (optional)
        """
        # Draw top-left scanner controls
        frame = self.draw_scanner_controls_panel(
            frame,
            current_angle=scanner_params.get('current_angle', 0),
            rotation_step=scanner_params.get('rotation_step', 30),
            current_session=scanner_params.get('current_session', 0),
            points_count=scanner_params.get('points_count', 0),
            current_mode=scanner_params.get('current_mode', 'Unknown'),
            curves_info=scanner_params.get('curves_info'),
            sensitivity_info=scanner_params.get('sensitivity_info')  # 🎨 NEW
        )
        
        # Draw message bar (between panels)
        frame = self.draw_message_bar(frame)
        
        # Draw bottom-right AI/quality panel
        frame = self.draw_ai_quality_panel(frame, ai_result)
        
        return frame
    
    def toggle_info_box(self):
        """Toggle scanner controls panel visibility."""
        self.info_box_visible = not self.info_box_visible
        return self.info_box_visible
    
    def toggle_ai_panel(self):
        """Toggle AI panel visibility."""
        self.ai_panel_visible = not self.ai_panel_visible
        return self.ai_panel_visible
    
    def toggle_terminal(self):
        """Toggle terminal output panel visibility."""
        self.terminal_visible = not self.terminal_visible
        return self.terminal_visible


# Example usage
if __name__ == "__main__":
    print("Use with laser_3d_scanner_advanced.py")

def scan_3d_points(project_dir=None):
    """Main 3D scanning function with live camera feed and laser detection."""
    
    # Global variables
    global roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2
    global roi_selecting, roi_start_x, roi_start_y
    global red_hue_min, red_hue_max, saturation_min, value_min
    global LASER_MIN_AREA, LASER_MAX_AREA
    global CANNY_THRESHOLD1, CANNY_THRESHOLD2
    global info_box_visible, ai_panel_visible, cartoon_mode
    global auto_capture_mode, auto_capture_countdown, auto_capture_count, auto_capture_target
    
    # Rotation tracking
    capture_metadata = {'angles': [], 'sessions': [], 'timestamps': []}
    current_session = 0
    current_angle = 0.0
    rotation_step = 30.0
    
    # Point cloud data
    points_3d = []
    points_colors = []
    point_angles = []
    point_sessions = []
    
    # 🎨 Sensitivity settings (NEW)
    curve_sample_rate = 5      # Curve point sampling rate (1 = all points, 5 = every 5th)
    corner_max_count = 100     # Maximum corners to capture per frame
    canny_threshold1 = 50      # Canny edge detection lower threshold
    canny_threshold2 = 150     # Canny edge detection upper threshold
    
    # Performance tracking
    profiler = PerformanceProfiler(window_size=30)
    
    # ... rest of function ...