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

raw_path = "/home/daq/LecroyMount/"

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

    #for i in range(len(time)):
    #    times = np.array(time[i][0])*10**9
    #    print(times[-1])
    #exit()

    # print(len(times))
    # print(len(clocks[0]))
    nGoodSamples = 5
    clock_timestamp = []
    #print(len(time[0][0]))
    #print(len(clocks[0]))
    #exit()
    for idx, clock in enumerate(clocks):
        times = np.array(time[idx][0])
        print(idx)
        amp_fraction = 20 # %
        risingX = []
        risingY = []
        minima = np.min(clock)
        maxima = np.max(clock)
        min_scale = np.abs(maxima - minima)/10.0
        check_fall = 0
        for k, s in enumerate(clock):
            if k == 0: continue
            if ((s - clock[k-1]) > min_scale): # and check_fall == 1:
                if len(risingY) > 0:
                    if (s < risingY[-1]):
                        if (len(risingY) == 1):
                            risingX = []
                            risingY = []
                            continue
                        else:
                            break
                risingX.append(times[k]*10**9)
                risingY.append(s)
                if check_fall == 0:
                    risingX = []
                    risingY = []
                    continue
            ##### DO NOT USE: ON IMPROVING #####
            elif (s - clock[k-1]) < -min_scale:
                check_fall = 1
            #    if len(risingY) > 0:
            #        if (s > risingY[-1]):
            #            if (len(risingY) == 1):
            #                risingX = []
            #                risingY = []
            #                continue
            #            else:
            #                break
            #   
            #    risingX.append(times[k]*10**9)
            #    risingY.append(s)
            #print(risingX)
            #print(risingY)
        risingX = np.array(risingX)
        risingY = np.array(risingY)
        print(risingX)
        print(risingY)
        popt, pcov = curve_fit(linear, risingX, risingY)
        amp = minima + np.abs(minima - maxima)*amp_fraction/100
        print(risingY)
        print(risingX)
        print(amp)
        clock_timestamp.append([(amp - popt[1]) / popt[0]])
        if (len(risingX) != 0) and (idx % 10 == 0) :
            print(len(times*10**9), len(clock))
            plt.plot(risingX, risingY, 'o')
            plt.plot(np.array(times)*(10**9), np.array(clock))
            plt.plot(risingX, linear(risingX, *popt))
            plt.savefig("./fit_results/clock_{0}.png".format(idx))
            plt.close()        # print(popt)
        # print((risingY*(20/100) - popt[1]) / popt[0])
    # tree["Clock"] = np.array(clock_timestamp)
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
        # print(data)
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
    base = "/home/daq/ETROC2_Test_Stand/ScopeHandler"
    # os.system("./reset_merging.sh")
    f_index = int(open(f"{base}/Lecroy/Acquisition/next_run_number.txt").read())
    print(f_index, "\n")
    prev_status = 0
    while True:
        reco_tree  = f"{base}/ScopeData/LecroyConverted/converted_run{f_index}.root"
        scope_tree = f"{base}/ScopeData/LecroyTimingDAQ/run_scope{f_index}.root"
        etroc_tree = f"{base}/ScopeData/ETROCData/output_run_{f_index}.root"
        reco_1 = (open(f"{base}/Lecroy/Conversion/merging.txt",                    "r").read() == "True")
        reco_2 = (path.isfile(reco_tree ))
        reco = reco_1 and reco_2
        scope_1 = (open(f"{base}/Lecroy/Acquisition/merging.txt",                  "r").read() == "True")
        scope_2 = (path.isfile(scope_tree))
        scope = scope_1 and scope_2
        etroc_1 = (open(f"/home/daq/ETROC2_Test_Stand/module_test_sw/merging.txt", "r").read() == "True")
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
            f = open(f"/home/daq/ETROC2_Test_Stand/module_test_sw/merging.txt", "w")
            f.write("False")
            f.truncate()
            f.close()
        time.sleep(1)
        prev_status = status
