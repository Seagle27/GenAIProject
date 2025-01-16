import os
import pandas as pd
import shutil


def separate_test_videos(csv_path, video_folder, output_folder):
    """
    Separate test videos from a directory based on a CSV file.

    Parameters:
    csv_path (str): Path to the CSV file containing video information.
    video_folder (str): Path to the folder containing video files.
    output_folder (str): Path to the folder where test videos will be copied.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create a set of test video IDs from the CSV
    test_video_ids = set(df[df['set'] == 'test']['ytid'])

    # Iterate over the files in the video folder
    for file in os.listdir(video_folder):
        if file.endswith('.mp4'):
            video_id = os.path.splitext(file)[0][:11]  # Get the file name without extension

            # Check if the video ID is in the test set
            if video_id in test_video_ids:
                source_path = os.path.join(video_folder, file)
                destination_path = os.path.join(output_folder, file)

                # Copy the file to the output folder
                shutil.copy(source_path, destination_path)
                print(f"Copied: {file}")


if __name__ == "__main__":
    # Example usage
    csv_path = r"C:\BGU\Year_4\GenAI\AudioToken\data\VGGSound\vggsound.csv"
    video_folder = r"C:\BGU\Year_4\GenAI\AudioToken\VGGSound\scratch\shared\beegfs\hchen\train_data\VGGSound_final" \
                   r"\video"
    output_folder = os.path.join(video_folder, 'test')

    separate_test_videos(csv_path, video_folder, output_folder)
