import sys, os, subprocess

# Check for the OneFormer repository
if not os.path.exists('./OneFormer'):
    subprocess.run(['git', 'clone', 'https://github.com/SHI-Labs/OneFormer'], check=True)

# Add OneFormer to the Python path for development purposes
sys.path.insert(0, os.path.abspath('./OneFormer'))