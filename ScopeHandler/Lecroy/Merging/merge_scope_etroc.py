import uproot
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from scipy.optimize import curve_fit
import awkward as ak
import glob
import time
import pdb
import os

raw_path = "/home/daq/LecroyMount/"  # PATH TO THE SCOPE RAW DATA FOLDER

def linear(x, a, b):
    return a*x + b

def find_cross_x(array, start, direction):
    # array = np.abs(array)
    minima = np.min(array)
    maxima = np.max(array)
    min_scale = np.abs(maxima - minima)/10.0
    max_scale = np.abs(maxima - minima)*9.0/10.0
    crossing_point = start
    # for i in range(len(array)):
    i = start
    while i < len(array):
        # print(array)
        if i != 0 and (array[i] - array[i - 1]) > min_scale and direction < 0 and (array[i] - min(array)) > min_scale and (array[i] - min(array)) < max_scale:
            crossing_point = i
            break
        if i != 0 and (array[i - 1] - array[i]) > min_scale and direction < 0 and (array[i] - min(array)) > min_scale and (array[i] - min(array)) < max_scale:
            crossing_point = i
            break
        else:
            i+=direction
    return crossing_point

def add_clock(tree):
    channel = tree['channel'].array()
    time = tree['time'].array()
    nSamples = len(time)
    clocks = channel[:,6]
    triggers = channel[:,7]
    times = np.array(time[:,0])*10**9
    clock = np.array(clocks)
    minima = np.tile(np.min(clock, axis=1).reshape(-1,1), (1, len(clock[0])))
    maxima = np.tile(np.max(clock, axis=1).reshape(-1,1), (1, len(clock[0])))
    amp_fraction = 20 # %
    amp = minima + np.abs(minima - maxima)*amp_fraction/100

    min_scale = np.abs(maxima - minima)/10.0

    clock_diff = np.diff(clock, append=0)
    clock_diff_mask = clock_diff > min_scale
    # true after indices
    check_prior_fall = clock_diff < -min_scale
    prior_indices = np.argmax(check_prior_fall, axis=1)

    prior_fall_mask = np.arange(check_prior_fall.shape[1]) >= prior_indices[:, None]

    global_mask = clock_diff_mask & prior_fall_mask

    times = np.where(global_mask, times, 0)
    clock = np.where(global_mask, clock, 0)
    # delete 0 values for each row
    times = ak.Array([sublist[sublist != 0] for sublist in times])
    clock = ak.Array([sublist[sublist != 0] for sublist in clock])

    print(times)
    time_slope = times[:,1] - times[:,0]
    clock_slope = clock[:,1] - clock[:,0]
    slope = clock_slope / time_slope
    ybias = clock[:,0] - slope*times[:,0]

    # calculate 20% of the amplitude
    amp = (minima + np.abs(minima - maxima)*amp_fraction/100)[:,0]
    clock_timestamp = np.array((amp - ybias) / slope)
    return clock_timestamp

def merge_trees(files, trees, output_file):
    # Read ROOT files and trees
    ts = [uproot.open(files[t])[tree] for t, tree in enumerate(trees)]
    print(ts)

    # Load data from trees .arrays()
    clock = add_clock(uproot.open(files[1])["pulse"])
    datas = [t.arrays() for t in ts]
    print(type(datas[0]))
    print(datas[1]["i_evt"])
    print(len(clock))
    datas.append(ak.Array({"Clock": clock}))

    # Merge the two datasets
    merged_data  = {}
    common_keys  = []
    other_keys_1 = []
    other_keys_2 = []

    for data in datas:
        for key in data.fields:
            if key not in merged_data.keys():
                merged_data[key] = data[key]

    print(merged_data.keys(), len(merged_data.keys()))

    # Create a new output file and write the merged tree
    with uproot.recreate(output_file) as output:
        output[trees[0]] = {key: merged_data[key] for key in merged_data.keys()}
        print(output[trees[0]].num_entries)


if __name__ == "__main__":
    # Replace these file names and tree names with your actual file names and tree names 
    # /home/daq/ETROC2_Test_Stand/ScopeHandler/Lecroy/Conversion/merging.txt
    base = "../../ScopeHandler"
    # os.system("./reset_merging.sh")
    f_index = int(open(f"{base}/Lecroy/Acquisition/next_run_number.txt").read())-1
    print(f_index, "\n")
    prev_status = 0
    while True:
        reco_tree  = f"{base}/ScopeData/LecroyConverted/converted_run{f_index}.root"
        scope_tree = f"{base}/ScopeData/LecroyTimingDAQ/run_scope{f_index}.root"
        etroc_tree = f"{base}/ScopeData/ETROCData/output_run_{f_index}.root"
        reco_1 = (open(f"{base}/Lecroy/Conversion/merging.txt",                    "r").read() == "True")
        reco_2 = (path.isfile(reco_tree))
        reco = reco_1 and reco_2
        scope_1 = (open(f"{base}/Lecroy/Acquisition/merging.txt",                  "r").read() == "True")
        scope_2 = (path.isfile(scope_tree))
        scope = scope_1 and scope_2
        etroc_1 = (open(f"../../module_test_sw/merging.txt", "r").read() == "True")
        etroc_2 = (path.isfile(etroc_tree))
        etroc = etroc_1 and etroc_2
        merged_file = f"{base}/ScopeData/LecroyMerged/run_{f_index}.root"
        status = sum([reco_1, reco_2, scope_1, scope_2, etroc_1, etroc_2, not path.isfile(merged_file)])

        if abs(status - prev_status) > 0:
            print( "                                  Flag      File")
            print(f"Acquisition from the scope done: {scope_1} {scope_2}")
            print(f"Acquisition from the KCU done:   {etroc_1} {etroc_2}")
            print(f"Conversion done:                 {reco_1} {reco_2}")
            print(f"Merged file wasn't created:      {not path.isfile(merged_file)}")
            print()

        if reco and scope and etroc: #and (not path.isfile(merged_file)):
            print("Secondary checking")
            print(reco)
            print(scope)
            print(etroc)
            print(not path.isfile(merged_file))
            print("\n")
            print(f"Reco data: {base}/ScopeData/LecroyConverted/converted_run{f_index}.root")
            print(f"Scope data: {base}/ScopeData/LecroyTimingDAQ/run_scope{f_index}.root")
            print(f"ETROC data: {base}/ScopeData/ETROCData/output_run_{f_index}.root")
            print(f"Merged data: {merged_file}")
            time.sleep(10)
            # try:
            merge_trees([reco_tree, scope_tree, etroc_tree], ["pulse", "pulse", "pulse"], merged_file)

                # print("Merging done!")
            # except Exception as error:
            #     print(repr(error))
            f_index = int(open(f"{base}/Lecroy/Acquisition/next_run_number.txt").read())
            print("\n", f_index, "\n")
            f = open(f"{base}/Lecroy/Conversion/merging.txt", "w")
            f.write("False")
            f.truncate()
            f.close()
            f = open(f"{base}/Lecroy/Acquisition/merging.txt", "w")
            f.write("False")
            f.truncate()
            f.close()
            f = open(f"../../module_test_sw/merging.txt", "w")
            f.write("False")
            f.truncate()
            f.close()
        time.sleep(1)
        prev_status = status
