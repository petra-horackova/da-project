import os
import time

import pandas as pd
import numpy as np


def get_df_cube(z, y, x, file_name):
    df = pd.read_csv(file_name)

    cube = np.zeros(shape=[z, y, x], dtype=float)
    for index, row in df.iterrows():
        cube[int(row["hour"])][int(row["lat_y"])][[int(row["lon_x"])]] = float(row["alerts"])

    return cube


def prepare_output_dir(custom_suffix):
    epoch_time = str(round(time.time()))
    dirname = 'outputs/sim_out_' + epoch_time + custom_suffix
    os.makedirs(dirname)
    print('creating directory: ', dirname)
    if not os.path.exists(dirname):
        raise Exception("output directory could not be created")
    print('success')
    return dirname
