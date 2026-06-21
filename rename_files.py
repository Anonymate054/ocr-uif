import os
import re
import unicodedata

def clean_name(name):
    # Separate the filename and extension if it's a file
    base, ext = os.path.splitext(name)
    
    # Normalize unicode to decompose accents (e.g., "Ó" -> "O" + accent mark)
    normalized = unicodedata.normalize('NFKD', base)
    ascii_base = normalized.encode('ascii', 'ignore').decode('utf-8')
    
    # Convert to lowercase
    lowercased = ascii_base.lower()
    
    # Replace any sequence of non-alphanumeric characters (except hyphens) with a single underscore
    cleaned = re.sub(r'[^a-z0-9\-]+', '_', lowercased)
    
    # Trim leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # If the base name becomes empty, fall back to "unnamed"
    if not cleaned:
        cleaned = "unnamed"
        
    # Reattach extension if it exists
    if ext:
        # Keep extension in lowercase
        return f"{cleaned}{ext.lower()}"
    return cleaned

def rename_contents(root_dir):
    print(f"Starting renaming in directory: {root_dir}")
    renamed_count = 0
    
    # Use topdown=False so we rename files/subfolders before their parent folders
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Rename files first
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_name = clean_name(filename)
            new_path = os.path.join(dirpath, new_name)
            
            if old_path != new_path:
                print(f"Renaming file: '{filename}' -> '{new_name}'")
                os.rename(old_path, new_path)
                renamed_count += 1
                
        # Rename directories
        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)
            new_name = clean_name(dirname)
            new_path = os.path.join(dirpath, new_name)
            
            if old_path != new_path:
                print(f"Renaming directory: '{dirname}' -> '{new_name}'")
                os.rename(old_path, new_path)
                renamed_count += 1
                
    print(f"Completed! Total items renamed: {renamed_count}")

if __name__ == "__main__":
    target_dir = "/home/lenovo/Documents/projects/ocr-uif/files"
    if os.path.exists(target_dir):
        rename_contents(target_dir)
    else:
        print(f"Error: Target directory {target_dir} does not exist.")
