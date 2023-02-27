import numpy as np
from sys import argv

params_num = int(argv[1])
accuracy = float(argv[2])


denominator = np.floor(np.log10(params_num)) + 1
power = (accuracy - 90)/100
final = (1/denominator)*(100**power)

print("final score:", final)
