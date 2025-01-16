import os
import subprocess
import pandas as pd
import shutil


def convert_mp4_to_wav(input_folder, output_folder, csv_path, labels, n):
    """
    Convert .mp4 files with specified labels in the CSV to .wav files using FFmpeg.

    Parameters:
    input_folder (str): Path to the folder containing .mp4 files.
    output_folder (str): Path to the folder where .wav files will be saved.
    csv_path (str): Path to the CSV file containing labels.
    labels (list): List of labels to filter MP4 files.
    n (int): Number of .mp4 files to convert.
    """
    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: The CSV file '{csv_path}' does not exist.")
        return

    # Load CSV and filter based on labels
    df = pd.read_csv(csv_path)
    filtered_files = df[df['class'].isin(labels)]['ytid'].tolist()

    if len(filtered_files) == 0:
        print("No matching MP4 files found for the specified labels.")
        return

    # Get the list of .mp4 files in the folder
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # Intersect the MP4 files with the filtered file list
    files_to_convert = [f for f in mp4_files if os.path.splitext(f)[0][:11] in filtered_files][:n]

    if len(files_to_convert) == 0:
        print("No matching MP4 files found in the input folder.")
        return

    # Convert each file
    for file in files_to_convert:
        # shutil.copyfile(os.path.join(input_folder, file), os.path.join(input_folder, 'labeled', file))
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.wav")

        try:
            subprocess.run([
                "ffmpeg", "-i", input_path, output_path, "-y"
            ], check=True)

            print(f"Converted: {file} -> {os.path.basename(output_path)}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {file}: {e}")


if __name__ == "__main__":
    video_folder = r"C:\BGU\Year_4\GenAI\AudioToken\VGGSound\scratch\shared\beegfs\hchen\train_data\VGGSound_final\video"
    audio_folder = r"C:\BGU\Year_4\GenAI\AudioToken\VGGSound\scratch\shared\beegfs\hchen\train_data\VGGSound_final\audio"
    csv_path = r"C:\BGU\Year_4\GenAI\AudioToken\data\VGGSound\vggsound.csv"
    labels = ["dog barking", "helicopter", "playing electric guitar", "sharpen knife", "alarm clock ringing"]
    convert_mp4_to_wav(video_folder, audio_folder, csv_path, labels, 3000)
