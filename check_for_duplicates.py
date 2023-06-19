from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

clean_docs_filepath = os.environ['PATH_TO_CLEAN_DOCS']

def count_files_in_directory(directory_path):
    return len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

# count files before deleting duplicates
print(f"Files before deleting duplicates: {count_files_in_directory(clean_docs_filepath)}")

def get_file_hash(filepath):
    # calculate SHA-1 hash of file
    hasher = hashlib.sha1()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

seen_hashes = set()
for filename in os.listdir(clean_docs_filepath):
    filepath = os.path.join(clean_docs_filepath, filename)
    file_hash = get_file_hash(filepath)

    if file_hash in seen_hashes:
        # This file is a duplicate, remove it
        os.remove(filepath)
    else:
        seen_hashes.add(file_hash)

# count files after deleting duplicates
print(f"Files after deleting duplicates: {count_files_in_directory(clean_docs_filepath)}")