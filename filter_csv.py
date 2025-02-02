import pandas as pd


def filter_and_sample_csv(csv_file: str, text_file: str, output_file: str = "filtered_output.csv", sample_size: int = 170):
    """
    Filters a CSV file based on labels from a text file and keeps only a random sample of 170 items per label.

    :param csv_file: Path to the CSV file.
    :param text_file: Path to the text file containing valid labels (one per line).
    :param output_file: Path to save the filtered and sampled CSV file (default: "filtered_output.csv").
    :param sample_size: Maximum number of rows to keep per label (default: 170).
    """
    # Read labels from the text file
    with open(text_file, "r", encoding="utf-8") as file:
        valid_labels = {line.strip() for line in file if line.strip()}

    # Read the CSV file (default assumes comma-separated values)
    df = pd.read_csv(csv_file)

    # Ensure the 'label' column exists in the CSV
    if "label" not in df.columns:
        raise ValueError("CSV file does not contain a 'label' column.")

    # Filter rows where the label is in the valid labels set
    filtered_df = df[df["label"].isin(valid_labels)]

    # Randomly sample up to `sample_size` items per label
    sampled_df = filtered_df.groupby("label").apply(lambda x: x.sample(n=min(len(x), sample_size), random_state=42)).reset_index(drop=True)

    # Save the final sampled CSV (default comma separator)
    sampled_df.to_csv(output_file, index=False)

    print(f"Filtered and sampled CSV saved as '{output_file}'.")


# Example usage
filter_and_sample_csv(r"C:\Users\tomer\Downloads\filtered_vggsound_top80.csv", "common_classes.txt")


