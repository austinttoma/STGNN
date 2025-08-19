import os
import shutil

def move_all_files_to_parent(folder_path):
    folder_path = os.path.abspath(folder_path)
    parent_path = os.path.dirname(folder_path)

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    moved_any = False

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_full_path = os.path.join(root, file)
            target_path = os.path.join(parent_path, file)

            if os.path.exists(target_path):
                print(f"Skipping '{file}' (already exists in parent folder)")
                continue

            shutil.move(file_full_path, target_path)
            print(f"Moved '{file}' to parent folder")
            moved_any = True

    if not moved_any:
        print("No files were moved.")
    else:
        print("Done moving all files.")


folder = '/media/volume/ADNI-Data/git/TabGNN/FinalDeliverables/data/DFC_Matrices/Updated_DFC_Matrices-20250818T113349Z-1-001'

move_all_files_to_parent(folder)
