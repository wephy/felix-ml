"""This script was used to clean the data on /storage/disqs/"""

import os
import numpy as np
from shutil import copyfile

current_path = os.getcwd()
data_loc = os.path.join(current_path, "Data")
target_loc = os.path.join(current_path, "cleaned_data")

ICSD_codes = set()

for datapoint in os.listdir(data_loc):
    entries = os.listdir(os.path.join(data_loc, datapoint))

    felix_output = entries[0]

    # CHECK DATA EXISTS, CONTINUE IF ISSUE
    try:
        pattern, structure_factors = os.listdir(
            os.path.join(data_loc, datapoint, felix_output)
        )
    except:
        no_data += 1
        continue

    # CHECK STRUCTURE FACTORS HAVE AMPLITUDE, CONTINUE IF ISSUE
    f = np.loadtxt(os.path.join(data_loc, datapoint, felix_output, structure_factors))
    if sum(f[:, 6]) == 0:
        no_amplitudes += 1
        continue

    # CHECK FOR DUPLICATES, CONTINUE IF ISSUE
    f = open(os.path.join(data_loc, datapoint, "felix.cif"))
    text = f.readlines()
    for line in text:
        if line[:19] == "_database_code_ICSD":
            ICSD_code = line.split(" ")[1][:-1]
    if ICSD_code in ICSD_codes:
        continue
    else:
        ICSD_codes.add(ICSD_code)

    # IF THE DATA IS OK
    # As the .hkl and .inp are not related to the chemical, they will stay as felix.*
    # however the .cif will become {ICSD_code}.cif, and likewise for the .bin and .txt
    # they will become {ICSD_code}_+0+0+0.bin and {ICSD_code}_structure_factors.txt
    os.mkdir(os.path.join(target_loc, ICSD_code))
    copyfile(
        os.path.join(data_loc, datapoint, "felix.hkl"),
        os.path.join(target_loc, ICSD_code, "felix.khl"),
    )
    copyfile(
        os.path.join(data_loc, datapoint, "felix.inp"),
        os.path.join(target_loc, ICSD_code, "felix.inp"),
    )
    copyfile(
        os.path.join(data_loc, datapoint, "felix.cif"),
        os.path.join(target_loc, ICSD_code, f"{ICSD_code}.cif"),
    )
    copyfile(
        os.path.join(data_loc, datapoint, felix_output, pattern),
        os.path.join(target_loc, ICSD_code, f"{ICSD_code}_+0+0+0.bin"),
    )
    copyfile(
        os.path.join(data_loc, datapoint, felix_output, structure_factors),
        os.path.join(target_loc, ICSD_code, f"{ICSD_code}_structure_factors.txt"),
    )
