extra_bias_values = [0.15]
bias_scores = [0, 0.15, 0.3]
thresholds = [0.12, 0.15, 0.18]
top_n_to_keep = [4, 5]
output_always = [True, False]

import itertools
import os

product = itertools.product(extra_bias_values, bias_scores, thresholds, top_n_to_keep, output_always)
for i, arg in enumerate(product):
    arg_string = " --extra_bias_intro_id2score " + str(arg[0]) +" --bias_id2score " + str(arg[1])\
                + " --threshold " + str(arg[2]) + " --top_n_to_keep " + str(arg[3]) + " --output_always " + str(arg[4])
    os.system('python3 0_parse_input.py ' + arg_string)

    os.system("rm -rf ./runtest/run1/Task1")
    os.system("cp -R ./runtest/Task1 ./runtest/run1/Task1")
    os.system("cd ./runtest && zip -r ../submissions/run" + str(i) +".zip ./run1/")
