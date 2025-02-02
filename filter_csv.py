import pandas as pd

def filter_csv_by_labels(csv_file: str, text_file: str, output_file: str = "filtered_vggsound_top30.csv"):
    """
    Filters a CSV file based on labels from a text file.

    :param csv_file: Path to the CSV file.
    :param text_file: Path to the text file containing valid labels (one per line).
    :param output_file: Path to save the filtered CSV file (default: "filtered_output.csv").
    """
    # Read labels from the text file
    with open(text_file, "r", encoding="utf-8") as file:
        valid_labels = {line.strip() for line in file if line.strip()}

    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter="\t")  # Assuming tab-separated values

    # Ensure the 'label' column exists in the CSV
    if "label" not in df.columns:
        raise ValueError("CSV file does not contain a 'label' column.")

    # Filter rows where the label is in the valid labels set
    filtered_df = df[df["label"].isin(valid_labels)]

    # Save the filtered CSV
    filtered_df.to_csv(output_file, index=False, sep="\t")

    print(f"Filtered CSV saved as '{output_file}'.")


# Example usage
filter_csv_by_labels(r"C:\Users\tomer\Downloads\filtered_vggsound_top80.csv", "common_classes.txt")


