import os

def delete_matching_files(input_dir, target_dir):
    # Collect all file names from input directory (recursively)
    input_files = set()

    for root, _, files in os.walk(input_dir):
        for file in files:
            input_files.add(file)

    # Traverse target directory and delete matching files
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file in input_files:
                target_path = os.path.join(root, file)
                os.remove(target_path)
                print(f"Deleted: {file}")

if __name__ == "__main__":
    input_directory = "data_split"
    target_directory = "klosterbestaende_writable_area"

    delete_matching_files(input_directory, target_directory)