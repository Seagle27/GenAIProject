import os
import cv2
import shutil

def check_and_move_videos(video_folder, output_folder, target_duration=10):
    """
    Checks video files in the given folder and moves those not matching the target duration
    to a specified output folder.

    Args:
        video_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder where mismatched videos will be moved.
        target_duration (int): The target duration (in seconds) for videos.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)

        # Ensure it's a file and has a valid video extension
        if not os.path.isfile(video_path) or not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            continue
        
        # Check the duration of the video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
        cap.release()

        if fps == 0:  # Handle invalid FPS or corrupted video files
            print(f"Skipping invalid video: {video_file}")
            continue
        
        duration = total_frames / fps  # Calculate duration in seconds

        # Check if the duration is not equal to the target duration
        if abs(duration - target_duration) > 0.1:  # Allow small tolerance for precision
            print(f"Moving {video_file}: duration = {duration:.2f} seconds")
            shutil.move(video_path, os.path.join(output_folder, video_file))
        else:
            print(f"Keeping {video_file}: duration = {duration:.2f} seconds")

# Example usage
video_folder = r"C:\BGU\Year_4\GenAI\AudioToken\VGGSound\scratch\shared\beegfs\hchen\train_data\VGGSound_final\video"
output_folder = r"C:\BGU\Year_4\GenAI\AudioToken\VGGSound\scratch\shared\beegfs\hchen\train_data\VGGSound_final\video\long"  # Replace with the path to store mismatched videos
check_and_move_videos(video_folder, output_folder)


