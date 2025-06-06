"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import sys, os, subprocess, shutil

# Define paths
repo_url = 'https://github.com/timbroed/CAFuser'
temp_clone_path = './CAFuser_temp'
target_folder = './cafuser'

# Clone only if the final folder doesn't exist
if not os.path.exists(target_folder):
    # Clone the repo into a temporary directory
    subprocess.run(['git', 'clone', repo_url, temp_clone_path], check=True)

    # Move only the desired subdirectory to the target location
    shutil.move(os.path.join(temp_clone_path, 'cafuser'), target_folder)

    # Clean up the temporary cloned repo
    shutil.rmtree(temp_clone_path)