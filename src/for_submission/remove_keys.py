import os
keep_files = ["C00-2123", "C04-1089", "I05-5011", "N06-2049", "P05-1004", "P98-1046", "P98-2143", "W03-0410"]
#keep_files = done_keys
for filename in os.listdir("./2018-evaluation-script/eval_dirs/res/Task1"):
    path = "./2018-evaluation-script/eval_dirs/res/Task1/"+filename
    if filename.replace(".csv", "") not in keep_files and os.path.isfile(path):
      pass
      print("removing : ", filename)
      os.remove(path)
    else:
      print("keeping : ", filename)