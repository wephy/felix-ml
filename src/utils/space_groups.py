import os
import numpy as np
import pandas as pd

data_loc = "data/FDP"

space_groups = dict()
space_groups_counts = dict()
n = len(os.listdir(data_loc))

for i, ICSD_code in enumerate(os.listdir(data_loc)):
    if i % (n // 100) == 0:
        print(f"{i / (n // 100)}% complete")
    cif = open(
        os.path.join(data_loc, ICSD_code, ICSD_code + ".cif"), "r"
    )
    lines = cif.readlines()
    found = False
    for line in lines:
        if "_symmetry_space_group_name_H-M " in line:
            space_group = line[line.find(" ")+1:]
            space_group = space_group.strip("'\n")
            space_groups[ICSD_code] = space_group
            # counts
            if space_group not in space_groups_counts:
                space_groups_counts[space_group] = 1
            else:
                space_groups_counts[space_group] += 1
            found = True
            break
    if found is True:
        continue
    for line in lines:
        if "space_group_name_H-M_alt " in line:
            space_group = line[line.find(" ")+1:]
            space_group = space_group.strip("'\n")
            space_groups[ICSD_code] = space_group
            # counts
            if space_group not in space_groups_counts:
                space_groups_counts[space_group] = 1
            else:
                space_groups_counts[space_group] += 1
            break

df = pd.DataFrame.from_dict(space_groups, orient="index", columns=["space_group"])
df.index.name = "ICSD_code"
df.to_csv('space_groups.csv', index=True)

df_counts = pd.DataFrame.from_dict(space_groups_counts, orient="index", columns=["count"])
df_counts.index.name = "space_group"
df_counts.to_csv('space_groups_counts.csv', index=True)
