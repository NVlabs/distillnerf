import os

# Set the directory containing the files
directory = '.'

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file name meets a condition to rename
    if 'dataset_config' in filename:
        # Create the new file name
        new_filename = filename.replace('dataset_config', 'dataset_config')
        
        # Create the full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed "{old_file}" to "{new_file}"')
