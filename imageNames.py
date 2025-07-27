import os

# Read all paths from your text file
with open('image_paths.txt', 'r', encoding='utf-8') as infile:
    lines = [line.strip() for line in infile if line.strip()]


# Extract the folder name (e.g., "DNA 2_00000110_result") from each path
folder_names = [os.path.basename(os.path.dirname(path)) for path in lines]

# Remove duplicates and sort the folder names alphabetically
unique_sorted_folders = sorted(set(folder_names))

# Write the sorted folder names to a new file
with open('folder_names.txt', 'w', encoding='utf-8') as outfile:
    for folder in unique_sorted_folders:
        outfile.write(f"{folder}\n")

print("Sorted folder names extracted to folder_names.txt")
