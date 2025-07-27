# import os

# # Path to the folder you showed (adjust this if needed)
# folder_path = "data/nagari/original/test/images"
# output_txt = "image_filenames_test.txt"

# # Get all .png files
# png_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".png"))

# # Write to a .txt file
# with open(output_txt, "w", encoding="utf-8") as f:
#     for filename in png_files:
#         f.write(filename + "\n")

# print(f"✅ Saved {len(png_files)} filenames to {output_txt}")


input_file = "image_filenames.txt"
output_file = "document_ids.txt"

seen = set()
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if "_textline_" in line:
            prefix = line.split("_textline_")[0]
            if prefix not in seen:
                fout.write(prefix + "\n")
                seen.add(prefix)

print(f"✅ Saved {len(seen)} unique document IDs to {output_file}")