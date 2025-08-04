import os
import shutil

copy_target = 100

def copy_files(file1, file2, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)  # create target folder if it doesn't exist

    src_file1 = os.path.join(source_dir, file1)
    src_file2 = os.path.join(source_dir, file2)

    dst_file1 = os.path.join(target_dir, "encoder.pt")
    dst_file2 = os.path.join(target_dir, "decoder.pt")

    shutil.copy2(src_file1, dst_file1)
    shutil.copy2(src_file2, dst_file2)

    print(f"✅ Copied:\n- {file1}\n- {file2}\nto {target_dir}")

# Example usage
if __name__ == "__main__":
    source_folder = "./vae_train/checkpoints"
    destination_folder = "./vae_custom/checkpoints"
    os.makedirs(destination_folder, exist_ok=True)
    file_name_1 = f"encoder_epoch_{copy_target}.pt"
    file_name_2 = f"decoder_epoch_{copy_target}.pt"

    copy_files(file_name_1, file_name_2, source_folder, destination_folder)
    print("Files copied successfully.")
