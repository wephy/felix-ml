import os
import numpy as np

flag_location = "//storage/disqs/felix-ML/ProjectSpace/VAE_000/Data"
 
img_array = np.load(os.path.join(flag_location, "1", "Output.npy"))

print(img_array)

from PIL import Image

im = Image.fromarray(img_array)
im.show()
