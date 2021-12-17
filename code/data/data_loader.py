# Author: Pranjali Parse
# Description: Function to load labelled data 

import pandas as pd

def data_loader(flag=0):
  # Path of the final data file: /content/gdrive/Shareddrives/CS685_Reformer_GEC/Data/FCE/fce_out/FinalData.csv
  data = pd.read_csv('drive/Shareddrives/CS685_Reformer_GEC/Data/FCE/fce_out/FinalDataUpdated.csv')
  
  # Convert string to list format
  def string_to_list(input_str):
    return input_str.strip('][').split(', ')

	# Convert string to tuple format
  def string_to_tuple(s):
      s = s.strip('][')
      return eval( "[%s]" % s )

  # If flag is 1, output returned for deletion tranformer else output returned for insertion transformer
  if flag == 1: 
    data["formatted"] = data["deletion"].apply(string_to_list)
    return data[["incorrect", "formatted"]].values.tolist()
  else:
    data["formatted"] = data["insertion"].apply(string_to_tuple)
    return data[["lcs", "formatted"]].values.tolist()

data = data_loader(flag = 0)
# Format of the output: List of lists
# For Example, Deletion: [["Hello How are yiu do", [0, 0, 0, 0, 0, 0, 0, 0, 1]], ["Good", [0, 0, 0, 0, 0, 1]]]
# For Example, Insertion: [["Hey I am good"], [(), (), (), ('n', ' ')], [" good"], [(), (), (), ('f', ' ')]]