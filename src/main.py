    # Tool Sequence Evaluation
	# 1) Bounding Box on Scene
	# 2) Check for Duplicate Hands
	# 2) CSV for Mediapipe at Setting
	# 3) CSV Data - Duplicate Frames, Outliers Marked, 


	# 1) Pre/Post Min, Max, Median Descriptive Stats
	# 2) Filtering Values
	# 3) Duplicate Frames - Marking
	# 4) Identify Key Frame Sections

import os
import sys
import pandas as pd

# Add the path to the modules directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the module
path_to_module = os.path.join(current_directory, '../modules/DIGITS_Model_Checker/src/')
path_to_module = os.path.normpath(path_to_module)  # Normalize the path

# Append the module directory to sys.path
sys.path.append(path_to_module)

from convert_mediapipe_index import convert_all_columns_to_friendly_name, convert_to_friendly_name, get_landmark_name


# Defin a main that loads a CSV file and converts the columns to user-friendly names
def main():
    # Load the CSV file
    filename = './data/saved_frame_data.csv'
    df = pd.read_csv(filename, sep=',')

    print(type(df))

    if df is None:
        print(f"Failed to load the CSV file {filename}. Exiting...")
        return
    
    print(df.shape)

    # Convert the columns to user-friendly names
    df.columns = convert_all_columns_to_friendly_name(df, [])

    # Save the DataFrame to a CSV file
    df.to_csv('./data/user_friendly_hand_landmarks.csv', sep=",", index=False)

if __name__ == "__main__":
    main()