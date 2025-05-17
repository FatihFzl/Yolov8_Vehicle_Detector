import cv2
import time
import yaml
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque

"""
Vehicle Detection and Tracking System

This project implements a real-time vehicle detection and tracking system using YOLOv8 and DeepSORT.
It processes video inputs to detect, classify, track, and count vehicles, providing visual feedback
and statistics about traffic flow.

Authors: [Fatih Burak Fazlioglu]
Institution: [Bahcesehir University]
Date: [2025-05-14]
"""

# Load YOLO model for vehicle detection
# The model has been trained on a custom dataset to detect vehicles in traffic scenes
model = YOLO("vehicle_detection/weights/best.pt")

# Load class mapping from YAML configuration
# This maps the numerical class IDs to human-readable vehicle type names
with open("vehicle_detection/class_map.yaml", "r") as f:
    class_names = yaml.safe_load(f)

# Configure DeepSORT tracker for vehicle tracking
# Parameters are optimized for vehicle tracking in traffic scenes:
# - max_age: How long to keep a track without detection before considering it lost
# - n_init: Minimum number of detections to initialize a track
# - max_iou_distance: Threshold for considering detections as matching
# - max_cosine_distance: Threshold for feature similarity matching
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_iou_distance=0.6,
    max_cosine_distance=0.2,
)

# Define color scheme for different vehicle classes in BGR format
# These colors provide good visual distinction between different vehicle types
class_colors = {
    'car': (0, 165, 255),      # Orange for cars
    'heavy_vehicle': (0, 0, 255),  # Red for heavy vehicles (trucks and buses)
    'van': (128, 0, 128),      # Purple for vans
    'motorcycle': (255, 0, 255), # Purple for motorcycles
    'bicycle': (255, 255, 0)   # Cyan for bicycles
}

# Initialize counters and tracking data structures
vehicle_count = defaultdict(int)  # Counts vehicles by class
tracked_ids = set()  # Set of tracked vehicle IDs to avoid double counting
tracked_positions = {}  # Dictionary to store position history for each vehicle
track_id_to_class = {}  # Maps each track_id to its assigned vehicle class
track_id_class_confidence = {}  # Stores confidence scores for class assignments
motion_threshold = 15   # Pixel distance threshold to consider a vehicle as having moved
start_time = datetime.now()  # Session start time for elapsed time calculation
passage_timestamps = defaultdict(lambda: deque(maxlen=100))  # Stores timestamps of vehicle passages by class
detection_threshold = 0.45  # Minimum confidence score to accept a detection

# Function to combine truck and bus counts into heavy_vehicle category
def merge_heavy_vehicles():
    """
    Merges the counts and timestamps of trucks and buses into a single 'heavy_vehicle' category.
    This is used to consolidate similar vehicle types that might be confused by the model.
    """
    global vehicle_count, passage_timestamps
    
    # Save the counts of trucks and buses
    truck_count = vehicle_count.get('truck', 0)
    bus_count = vehicle_count.get('bus', 0)
    
    # If there are any trucks or buses, merge them
    if truck_count > 0 or bus_count > 0:
        # Add the counts together in the heavy_vehicle category
        vehicle_count['heavy_vehicle'] = truck_count + bus_count
        
        # Remove the old categories
        if 'truck' in vehicle_count:
            del vehicle_count['truck']
        if 'bus' in vehicle_count:
            del vehicle_count['bus']
        
        # Merge the timestamp queues
        truck_timestamps = list(passage_timestamps.get('truck', []))
        bus_timestamps = list(passage_timestamps.get('bus', []))
        
        # Create a new combined queue
        heavy_vehicle_timestamps = deque(sorted(truck_timestamps + bus_timestamps), maxlen=100)
        passage_timestamps['heavy_vehicle'] = heavy_vehicle_timestamps
        
        # Remove the old timestamp queues
        if 'truck' in passage_timestamps:
            del passage_timestamps['truck']
        if 'bus' in passage_timestamps:
            del passage_timestamps['bus']

# Calculate traffic density based on vehicle counts in the last minute
def calculate_traffic_density():
    """
    Calculates current traffic density by counting vehicles that passed
    in the last 60 seconds. Returns a classification as LOW, MEDIUM, or HIGH.
    
    Returns:
        str: Traffic density classification
    """ 
    now = datetime.now()
    one_minute_ago = now - timedelta(seconds=60)
    total_passed = sum(
        sum(1 for ts in timestamps if ts > one_minute_ago)
        for timestamps in passage_timestamps.values()
    )

    if total_passed < 45:
        return "LOW"
    elif total_passed < 75:
        return "MEDIUM"
    else:
        return "HIGH"

# Count total vehicles that passed in the last 60 seconds
def count_last_minute_vehicles():
    """
    Counts the total number of vehicles that passed in the last 60 seconds
    across all vehicle classes.
    
    Returns:
        int: Total vehicle count in the last minute
    """
    now = datetime.now()
    one_minute_ago = now - timedelta(seconds=60)
    return sum(
        sum(1 for ts in timestamps if ts > one_minute_ago)
        for timestamps in passage_timestamps.values()
    )

# Draw a semi-transparent panel for displaying information on the video frame
def draw_transparent_panel(image, x, y, w, h, alpha=0.7, color=(20, 20, 20)):
    """
    Creates a semi-transparent panel on the image for displaying text information.
    Handles edge cases where panel might exceed image boundaries.
    
    Args:
        image: The frame to draw on
        x, y: Top-left corner coordinates
        w, h: Width and height of the panel
        alpha: Transparency factor (0-1)
        color: Panel background color in BGR
        
    Returns:
        image: The modified image with the panel
    """
    # Check panel boundaries against image dimensions
    h_img, w_img, _ = image.shape
    x_end = min(x + w, w_img)
    y_end = min(y + h, h_img)
    
    if x >= w_img or y >= h_img:
        return image
    
    sub_img = image[y:y_end, x:x_end]
    overlay = np.ones(sub_img.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)
    
    # Add gradient effect for more professional appearance
    for i in range(overlay.shape[0]):
        alpha_row = alpha - (i / overlay.shape[0]) * 0.1
        cv2.addWeighted(overlay[i:i+1, :], alpha_row, sub_img[i:i+1, :], 1-alpha_row, 0, sub_img[i:i+1, :])
    
    # Add a thin line at the panel edge for better visual definition
    cv2.rectangle(image, (x, y), (x_end-1, y_end-1), (100, 100, 100), 1)
    
    return image

# FPS calculator class for performance measurement
class FPS:
    """
    Calculates and maintains the frames per second rate of video processing.
    Uses a rolling window of timestamps for more accurate measurement.
    """
    def __init__(self, avg_frames=30):
        """
        Initialize the FPS calculator with a specified window size.
        
        Args:
            avg_frames: Number of frames to consider for FPS calculation
        """
        self.timestamps = deque(maxlen=avg_frames)
        self.start_time = time.time()
    
    def update(self):
        """Record the timestamp of a new frame."""
        self.timestamps.append(time.time())
    
    def get_fps(self):
        """
        Calculate the current FPS based on recorded timestamps.
        
        Returns:
            float: Calculated FPS value, or 0 if insufficient data
        """
        if len(self.timestamps) < 2:
            return 0
        
        # Calculate based on the difference between the first and last timestamps
        time_diff = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / time_diff if time_diff > 0 else 0

# Function to add information overlay to the video frame
def add_info_overlay(frame, vehicle_count, density, fps):
    """
    Adds an information panel to the video frame showing vehicle counts,
    traffic density, FPS, and other relevant statistics.
    
    Args:
        frame: The video frame to add overlay to
        vehicle_count: Dictionary of vehicle counts by class
        density: Current traffic density classification
        fps: Current frames per second value
        
    Returns:
        frame: The modified frame with information overlay
    """
    # Panel width and height - Compact panel height
    panel_width = 300
    total_items = len(vehicle_count) if vehicle_count else 0
    
    # Calculate panel height based on content
    base_height = 120  # For FPS, start time, last 60s, traffic density and title
    item_height = 25   # For each vehicle type
    percentage_area = 100  # Area for percentage and bar graph - increased further to 100
    
    panel_height = base_height + (total_items * item_height) + percentage_area
    max_panel_height = frame.shape[0] - 40
    panel_height = min(panel_height, max_panel_height)

    panel_x = 20
    panel_y = 20
    
    # Font settings for text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    small_font_scale = 0.5  # Small font scale
    font_color = (255, 255, 255)  # White
    thickness = 1
    
    # Draw semi-transparent panel as background for information
    frame = draw_transparent_panel(frame, panel_x, panel_y, panel_width, panel_height)
    
    # Panel content positioning
    content_x = panel_x + 15
    content_y = panel_y + 30
    
    # Start time information
    cv2.putText(frame, f"Start Time: {start_time.strftime('%H:%M')}", 
                (content_x, content_y), font, font_scale, font_color, thickness)
    content_y += 25  # Compact spacing
    
    # Vehicles passed in the last 60 seconds
    last_minute_count = count_last_minute_vehicles()
    cv2.putText(frame, f"Last 60s Vehicles: {last_minute_count}", 
                (content_x, content_y), font, font_scale, font_color, thickness)
    content_y += 25  # Compact spacing
    
    # Traffic density with color-coding based on level
    density_color = (0, 255, 0) if density == "LOW" else (0, 165, 255) if density == "MEDIUM" else (0, 0, 255)
    cv2.putText(frame, f"Traffic Density: {density}", 
                (content_x, content_y), font, font_scale, density_color, thickness)
    content_y += 25  # Compact spacing
    
    # FPS information for performance monitoring
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (content_x, content_y), font, font_scale, font_color, thickness)
    content_y += 25  # Compact spacing
    
    # Vehicle counts title
    cv2.putText(frame, f"Vehicle Counts:", 
                (content_x, content_y), font, font_scale, font_color, thickness)
    content_y += 25
    
    # Display count for each vehicle type with their respective colors
    indent_x = content_x + 15
    # For sorted display
    for cls in sorted(vehicle_count.keys()):
        if content_y > panel_y + panel_height - 30:
            break  # Exit if exceeding panel height
        
        count = vehicle_count[cls]
        color = class_colors.get(cls, (255, 255, 255))
        cv2.putText(frame, f"- {cls.capitalize()}: {count}", 
                    (indent_x, content_y), font, font_scale, color, thickness)
        content_y += 25
    
    # Mini bar graphs showing percentage distribution of vehicle types
    total_vehicles = sum(vehicle_count.values())
    if total_vehicles > 0:
        graph_start_x = panel_x + 15
        graph_start_y = content_y + 5
        graph_width = panel_width - 30
        graph_height = 10
        
        # Background for bar graph
        cv2.rectangle(frame, (graph_start_x, graph_start_y), 
                      (graph_start_x + graph_width, graph_start_y + graph_height), 
                      (50, 50, 50), -1)
        
        # Draw a bar segment for each vehicle class proportional to its count
        x_pos = graph_start_x
        percentage_text = ""  # Text to show percentage breakdown
        
        for cls in sorted(vehicle_count.keys()):
            count = vehicle_count[cls]
            ratio = count / total_vehicles
            bar_width = int(graph_width * ratio)
            if bar_width < 2 and count > 0:
                bar_width = 2  # Minimum visible width
                
            color = class_colors.get(cls, (255, 255, 255))
            
            cv2.rectangle(frame, (x_pos, graph_start_y), 
                         (x_pos + bar_width, graph_start_y + graph_height), 
                         color, -1)
            
            # Add to percentage text with first letter of class as identifier
            percentage_text += f"{cls[:1].upper()}:{int(ratio*100)}% "
            
            x_pos += bar_width
        
        # Total vehicle count
        cv2.putText(frame, f"Total: {total_vehicles}", 
                   (graph_start_x, graph_start_y + graph_height + 15), 
                   font, font_scale, font_color, thickness)
        
        # Show percentage values in one line below the graph
        pct_y = graph_start_y + graph_height + 30
        cv2.putText(frame, percentage_text.strip(), (graph_start_x, pct_y), 
                    font, small_font_scale, font_color, 1)
        
    return frame

# Check if a vehicle has moved sufficiently to be counted
def check_movement(track_id, current_center):
    """
    Determines if a tracked vehicle has moved enough from its previous position
    to be considered in motion. This helps filter out false positives and
    stationary objects.
    
    Args:
        track_id: Unique identifier for the tracked vehicle
        current_center: Current (x,y) center position of the vehicle
        
    Returns:
        bool: True if the vehicle has moved significantly, False otherwise
    """
    global tracked_positions
    prev_center = tracked_positions.get(track_id)
    moved = False
    
    if prev_center:
        dx = abs(current_center[0] - prev_center[0])
        dy = abs(current_center[1] - prev_center[1])
        if dx + dy > motion_threshold:
            moved = True
    
    # Always update position for next comparison
    tracked_positions[track_id] = current_center
    return moved

# Process each video frame for vehicle detection and tracking
def process_frame(frame):
    """
    Main processing function that detects and tracks vehicles in a video frame.
    Applies YOLO detection and DeepSORT tracking, then visualizes results.
    
    Args:
        frame: The video frame to process
        
    Returns:
        frame: The processed frame with detection boxes and information
    """
    # Run YOLO model on frame with tracking enabled
    results = model.track(frame, persist=True, conf=detection_threshold)
    if results[0].boxes.id is None:
        return frame

    # Extract detection results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # Tracking IDs
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Process each detected vehicle
    for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
        x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
        class_name = class_names.get(class_id, f"class_{class_id}")  # Get class name from ID
        
        # Combine 'truck' and 'bus' into 'heavy_vehicle'
        if class_name in ['truck', 'bus']:
            class_name = 'heavy_vehicle'
        
        # Ensure class consistency using track_id, giving preference to higher confidence detections
        if track_id in track_id_to_class:
            # If we've seen this vehicle before
            current_class = track_id_to_class[track_id]
            current_conf = track_id_class_confidence.get(track_id, 0.0)
            
            # Special case for trucks and buses - they should be treated as heavy_vehicle
            if current_class in ['truck', 'bus']:
                current_class = 'heavy_vehicle'
            
            # Update class if new detection has significantly higher confidence
            if conf > current_conf + 0.15:  # At least 15% higher confidence to switch class
                track_id_to_class[track_id] = class_name
                track_id_class_confidence[track_id] = conf
                class_name = track_id_to_class[track_id]
            else:
                # Otherwise use the previously assigned class
                class_name = current_class
        else:
            # First time seeing this track_id, save its class
            track_id_to_class[track_id] = class_name
            track_id_class_confidence[track_id] = conf
            
        # Calculate center point of the vehicle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_center = (center_x, center_y)

        # Check if vehicle has moved enough to be counted
        moved = check_movement(track_id, current_center)

        # Count new vehicles passing through the frame
        if moved and track_id not in tracked_ids:
            tracked_ids.add(track_id)
            vehicle_count[class_name] += 1
            passage_timestamps[class_name].append(datetime.now())
            # Draw indicator for newly counted vehicle
            cv2.circle(frame, current_center, 10, (0, 255, 0), -1)  # Green dot
            cv2.circle(frame, current_center, 10, (255, 255, 255), 2)  # White outline

        # Format confidence percentage for display
        confidence_pct = int(conf * 100)
        label = f"{class_name} ({confidence_pct}%)"
        
        # Get color specific to this vehicle class
        box_color = class_colors.get(class_name, (0, 255, 255))  # Default color is yellow
        
        # Draw bounding box and label with professional appearance
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add a background behind the label text for better readability
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0] + 5, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)  # Black text on colored background
                    
        # Add corner markers for professional appearance
        mark_len = 7
        # Top left corner
        cv2.line(frame, (x1, y1), (x1 + mark_len, y1), box_color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + mark_len), box_color, 2)
        # Top right corner
        cv2.line(frame, (x2, y1), (x2 - mark_len, y1), box_color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + mark_len), box_color, 2)
        # Bottom left corner
        cv2.line(frame, (x1, y2), (x1 + mark_len, y2), box_color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - mark_len), box_color, 2)
        # Bottom right corner
        cv2.line(frame, (x2, y2), (x2 - mark_len, y2), box_color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - mark_len), box_color, 2)

    return frame

# Resize frame to improve processing performance with large images
def resize_frame(frame, target_width=1280):
    """
    Resizes the input frame to a target width while maintaining aspect ratio.
    This improves processing performance for high-resolution videos.
    
    Args:
        frame: Input video frame
        target_width: Target width in pixels
        
    Returns:
        frame: Resized frame
    """
    height, width = frame.shape[:2]
    if width > target_width:
        ratio = target_width / width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (target_width, new_height))
    return frame

# Add timestamp to the video frame
def add_timestamp(frame):
    """
    Adds current date and time to the bottom right corner of the frame.
    Useful for logging when events occurred in the video.
    
    Args:
        frame: Video frame to add timestamp to
        
    Returns:
        frame: Frame with timestamp added
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    h, w = frame.shape[:2]
    # Show date/time in the bottom right
    cv2.putText(frame, timestamp, (w - 200, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame

# Main function to handle video processing
def main():
    """
    Main function that handles opening video sources, processing frames,
    and saving the results if requested. Supports command-line arguments
    for specifying input video and output file.
    """
    # Set up command line argument parsing for flexible input/output options
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Vehicle Detection and Tracking")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    args = parser.parse_args()
    
    # If the video path (args.video) is not specified from the command line,
    # the default video file "traffic_video.mp4" will be used.
    # This is a sample video located in the root directory of the project.
    # Users who want to use a different video should run the script as follows:
    # python vehicle_detection_tracking.py --video "your_video.mp4"
    video_path = args.video
    if not video_path:
        # Default video path if none provided
        video_path = "video_traffic.mp4"
    
    # Check if specified video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
        
    # Open video capture device
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Initialize FPS counter for performance tracking
    fps_counter = FPS()
    
    # Get video properties for processing and output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Configure output video settings if requested
    # The output video can be specified with the --output parameter
    # For example: python vehicle_detection_tracking.py --video input.mp4 --output result.avi
    save_output = True if args.output else False
    output_path = args.output if args.output else "processed_traffic.avi"
    
    if save_output:
        # Video writer - use AVI format and XVID codec for compatibility
        # AVI format is chosen for better compatibility across platforms
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if video writer was successfully initialized
        if not out.isOpened():
            print(f"Error: VideoWriter couldn't be opened. Trying with AVI format.")
            # Try creating with explicit AVI extension if there were format issues
            output_path = output_path if output_path.endswith('.avi') else output_path.rsplit('.', 1)[0] + '.avi'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Error: VideoWriter still couldn't be opened. Video won't be saved.")
                save_output = False
            else:
                print(f"Output video will be saved to: {output_path}")
        else:
            print(f"Output video will be saved to: {output_path}")
    
    # Print processing information for monitoring
    print(f"Processing video: {fps} FPS, Resolution: {width}x{height}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    frame_count = 0
    
    # Main video processing loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # End of video or error reading frame
            
        frame_count += 1
        
        # Process every 2nd frame for better performance
        # This optimization helps maintain real-time processing speed
        # without significantly affecting detection quality
        if frame_count % 2 == 0:
            # Update FPS calculation
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # Process the current frame with all detection and visualization steps
            processed_frame = resize_frame(frame.copy())  # Preserve original frame
            processed_frame = process_frame(processed_frame)  # Detect and track vehicles
            merge_heavy_vehicles()  # Merge truck and bus counts into heavy_vehicle
            density = calculate_traffic_density()  # Calculate current traffic density
            processed_frame = add_info_overlay(processed_frame, vehicle_count, density, current_fps)  # Add stats panel
            processed_frame = add_timestamp(processed_frame)  # Add timestamp
            
            # Save output video if requested
            if save_output and out.isOpened():
                # Resize processed frame back to original size for consistent output
                if processed_frame.shape[:2] != (height, width):
                    processed_frame = cv2.resize(processed_frame, (width, height))
                try:
                    # Write frame to output video
                    out.write(processed_frame)
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    save_output = False
            
            # Display the processed frame in a window
            cv2.imshow("Vehicle Detection & Tracking", processed_frame)
            # Allow user to exit by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    # Clean up resources
    cap.release()
    if save_output and out.isOpened():
        out.release()
        print(f"Video saved successfully to {output_path}")
    cv2.destroyAllWindows()
    
    # Print summary of processing results
    print("\nProcessing completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total vehicles detected: {sum(vehicle_count.values())}")
    for vehicle_type, count in sorted(vehicle_count.items()):
        print(f"  - {vehicle_type.capitalize()}: {count}")
    print(f"Output saved to: {output_path}" if save_output else "No output saved")

# Entry point for the script
if __name__ == "__main__":
    main()
