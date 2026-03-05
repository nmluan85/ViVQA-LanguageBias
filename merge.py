import hashlib
from pathlib import Path
import shutil

def get_file_hash(file_path):
    """Generates an MD5 hash to identify unique image content."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def merge_images_safely(source_dirs, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    seen_hashes = set()
    count = 0

    for source in source_dirs:
        src_path = Path(source)
        # Supports common image formats
        for img_file in [f for f in src_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.safetensors']]:
            
            file_hash = get_file_hash(img_file)
            
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                
                # Define destination path
                dest_file = out_path / img_file.name
                
                # If filename exists but hash is different, rename the copy
                if dest_file.exists():
                    dest_file = out_path / f"from_{src_path.name}_{img_file.name}"
                
                # Use copy2 to preserve metadata (dates, etc.)
                shutil.copy2(img_file, dest_file)
                count += 1

    print(f"Done! {count} unique files copied to '{output_dir}'.")
    print("Original folders A and B are unchanged.")

# --- Run the function ---
# You can add as many folders to this list as you like
merge_images_safely(['D:/Thesis/Dataset/ViVQA/drive-download-20220309T020508Z-001/test', 'D:/Thesis/Dataset/ViVQA/drive-download-20220309T020508Z-001/train'], 'data\images')