import nltk
import os

# Path where nltk_data should be downloaded
nltk_data_path = '/opt/render/nltk_data'

# Ensure the directory exists
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Append the path to nltk.data
nltk.data.path.append(nltk_data_path)

# Download 'punkt'
nltk.download('punkt', download_dir=nltk_data_path)
