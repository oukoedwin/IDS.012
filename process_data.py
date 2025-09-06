import os
import zipfile
import shutil
import pandas as pd

data_dir = "./temp"
combined_csv_path = os.path.join(data_dir, "combined.csv")

all_dfs = []
master_columns = None  

# First and only pass: extract, scan headers, and read data immediately
for zipped_folder in sorted(os.listdir(data_dir)):
    if zipped_folder == ".DS_Store":
        continue

    zipped_folder_path = os.path.join(data_dir, zipped_folder)

    folder_name = zipped_folder.split(".")[0]

    folder_path = os.path.join(data_dir, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    with zipfile.ZipFile(zipped_folder_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

    os.remove(zipped_folder_path)

    # Read all CSVs inside
    for csv_file in sorted(os.listdir(folder_path)):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(folder_path, csv_file)
            # df = pd.read_csv(csv_path, skiprows=6, header=4)
            df = pd.read_csv(
                csv_path,
                header=4,     
                skiprows=[5]
            )
            if master_columns is None:
                master_columns = df.columns.tolist()  # set column order
            else:
                new_columns = [col for col in df.columns if col not in master_columns]
                if new_columns:
                    master_columns.extend(new_columns)  

            df = df.reindex(columns=master_columns)
            all_dfs.append(df)

    shutil.rmtree(folder_path)

# Combine and save
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined CSV saved to {combined_csv_path}")
