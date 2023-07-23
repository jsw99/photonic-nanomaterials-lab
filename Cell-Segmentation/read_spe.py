import numpy as np
import matplotlib.pyplot as plt
import spe2py as spe

spe_tools = spe.load()

frame_data = spe_tools.file.data[0][0]