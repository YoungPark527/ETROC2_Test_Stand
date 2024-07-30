import uproot
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import awkward as ak
import argparse
import glob
import time
import pdb
import os
import sys

raw_path = "/home/daq/LecroyMount/"  # PATH TO THE SCOPE RAW DATA FOLDER
d = 1.2 # max(signal[s]) # (max(signal[s]) - min(signal[s]))
clk_bound_max = 0.7 # V
clk_bound_min = 0.2 # V
clk_threshold = 0.6 # V

argument_holder = argparse.ArgumentParser(description = "Argument parser")
argument_holder.add_argument("--plotting", action="store_true", default = False, help = "Plot the fits of the clock waveform.")
argument_holder.add_argument("--run_number", action="store", help = "The run number.", type = int)

args = argument_holder.parse_args()

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

def linear_fit_rising_edge(signal, time_axis, frequency, sampling_rate, which_peak = "first"):
    # Calculate the time period
    # Provide frequency in Herz
    time_period = (1 / frequency) * 10**9
    # print(time_period)
    half_period_samples = int((time_period / 2) * sampling_rate)
    
    # rising_edges_i = []
    maxes = []
    mins = []
    slopes = []
    biases = []
    rising_edges_mV = []
    rising_edges_t  = []
    N_peaks_before_max = []
    N_peaks_before_min = []
    N_peaks_after = []
    peak_index_min = []
    peak_index_max = []
    n_points_rising_edge = []
    n_fit_points = []

    for s in range(len(signal)): # /////////////////////////////////////////////////////////////// LOOP OVER THE EVENTS (WAVEFORMS)
        print(f"{s} / {len(signal)}")
        bias  = []
        slope = []
        
        # Find peaks
        peaks = []
        
        amp_step   = d * 1 / 100
        height_min = clk_bound_min # d * prop_min / 100 # min(signal[s]) + d * prop_min / 100
        height_max = clk_bound_max # d - d * prop_max / 100 # max(signal[s]) - d * prop_max / 100
        
        p_min, _ = find_peaks(-signal[s], height = (-1*height_min, -1*min(signal[s])), distance=int(len(signal[s])/4))
        p_max, _ = find_peaks(-signal[s], height = (-1*max(signal[s]), -1*height_max), distance=int(len(signal[s])/4))
        
        N_peaks_before_max.append(len(p_max))
        N_peaks_before_min.append(len(p_min))
        if p_max[0] < p_min[0]:
            # peaks.append(p_max[1])
            p_max = p_max[1::]
        if p_max[-1] < p_min[-1]:
            # peaks.append(p_max[0])
            p_min = p_min[0:-1]
        # print(p_min)
        # print(p_max)
        N_peaks_after.append(len(p_max))
        
        amp_step = d * 1 / 100
        
        r_e_start = []
        r_e_end   = []
        rising_edge_mV = []
        rising_edge_t = []
        n_f_points = []
        
        # print(f"The length of the max and the min: {len(p_max)}, {len(p_min)}.")
        for p in range(len(p_max)): # /////////////////////////////////////////////////////////////// LOOP OVER THE PEAKS
            # height_min = d * prop_min / 100 # signal[s][p_min][p] + d[p] * prop_min / 100
            # height_max = d - d * prop_max / 100 # signal[s][p_max][p] - d[p] * prop_max / 100
            start = p_min[p]
            end   = p_max[p]
            
            rising_edge_start = start
            for i in range(start, end - 1):
                if signal[s][i] < signal[s][i+1] and abs(signal[s][i] - signal[s][i+1]) > amp_step and signal[s][i] >= height_min:
                    rising_edge_start = i
                    break
            r_e_start.append(rising_edge_start)
            rising_edge_end = rising_edge_start
            for i in range(rising_edge_start, end):
                if signal[s][i] >= height_max:
                    rising_edge_end = i
                    break
            r_e_end.append(rising_edge_end)
            
            rising_edge_mV.append(signal   [s][r_e_start[p]:r_e_end[p]])
            rising_edge_t .append(time_axis[s][r_e_start[p]:r_e_end[p]])
            # popt, _ = curve_fit(linear, [rising_edge_t[p][0], rising_edge_t[p][-1]], [rising_edge_mV[p][0], rising_edge_mV[p][-1]])
            popt, _ = curve_fit(linear, rising_edge_t[p], rising_edge_mV[p])
            n_f_points.append(len(signal[s][r_e_start[p]:r_e_end[p]]))
            slope.append(popt[0])
            bias.append(popt[1])
        rising_edges_mV.append(list(rising_edge_mV))
        rising_edges_t.append(list(rising_edge_t))
        slopes.append(slope)
        biases.append(bias)
        maxes.append(r_e_end)
        mins.append(r_e_start)
        peak_index_min.append(p_min)
        peak_index_max.append(p_max)
        n_fit_points.append(n_f_points)

    return slopes, biases, rising_edges_mV, rising_edges_t, N_peaks_after, peak_index_min, peak_index_max, n_fit_points

n_of_e = 10000

def add_clock_old(tree):
    correction_offset = tree['timeoffsets'].array()[0:n_of_e]
    channel = tree['channel'].array()[0:n_of_e]
    time = tree['time'].array()[0:n_of_e]
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

    time_slope = times[:,1] - times[:,0]
    clock_slope = clock[:,1] - clock[:,0]
    slope = clock_slope / time_slope
    ybias = clock[:,0] - slope*times[:,0]

    # calculate 20% of the amplitude
    amp = (minima + np.abs(maxima - minima)*amp_fraction/100)[:,0]
    clock_timestamp = ((amp - ybias) / slope) + correction_offset[:,6]*10**9
    return clock_timestamp

def add_clock(tree, frequency = 40*10**6, plotting = False): # Hz
    correction_offset = tree['timeoffsets'].array()[0:n_of_e]
    channel = tree['channel'].array()[0:n_of_e]
    time = tree['time'].array()[0:n_of_e]
    nSamples = len(time)
    clocks = channel[:,6]
    triggers = channel[:,7]
    times = np.array(time[:,0])*10**9
    clock = np.array(clocks)
    slopes, biases, rising_edges_mV, rising_edges_t, n_peaks_after_truncation, peak_index_min, peak_index_max, n_fit_points = linear_fit_rising_edge(clock, times, frequency, 1)
    clock_timestamps_10 = []
    clock_timestamp_differences_10 = []
    amps_10 = []
    min_rising_edge_t = []
    max_rising_edge_t = []
    min_rising_edge_mV = []
    max_rising_edge_mV = []
    for s in range(len(slopes)): # loop over the events
        print(f"{s} / {len(slopes)}")
        t_stamps_10 = []
        t_stamps_30 = []
        t_stamps_50 = []
        t_stamps_70 = []
        a_10 = []
        a_30 = []
        a_50 = []
        a_70 = []
        mins_t = []
        maxes_t = []
        mins_mV = []
        maxes_mV = []
        clock_timestamp_difference_10 = []

        if plotting:
            plt.figure(figsize = (15, 12))
            plt.plot(times[s], clock[s])
            plt.scatter(times[s], clock[s], marker = "o", color = "orange")

        for i in range(len(slopes[s])): # loop over the peaks
            minima = min(rising_edges_mV[s][i])
            maxima = max(rising_edges_mV[s][i])
            min_index = np.argmin(rising_edges_mV[s][i])
            max_index = np.argmax(rising_edges_mV[s][i])
            mins_t.append(rising_edges_t[s][i][min_index])
            maxes_t.append(rising_edges_t[s][i][max_index])
            mins_mV.append(rising_edges_mV[s][i][min_index])
            maxes_mV.append(rising_edges_mV[s][i][max_index])
            height_min = clk_bound_min # d * prop_min / 100 # min(signal[s]) + d * prop_min / 100
            height_max = clk_bound_max # d - d * prop_max / 100 # max(signal[s]) - d * prop_max / 100
            amp_10 = clk_threshold # height_min + np.abs(height_max - height_min) * 10 / 100.0
            a_10.append(amp_10)
            clock_timestamp_10 = ((amp_10 - biases[s][i]) / slopes[s][i]) + correction_offset[:,6][s]*10**9
            t_stamps_10.append(clock_timestamp_10)

            if plotting:
                plt.plot(times[s], [minima]*len(clock[s]), "--", color="blue")
                plt.plot(times[s], [maxima]*len(clock[s]), "--", color="red")
                plt.plot(times[s][peak_index_min[s]], clock[s][peak_index_min[s]], "*", color="blue", markersize = 20)
                plt.plot(times[s][peak_index_max[s]], clock[s][peak_index_max[s]], "*", color="red", markersize = 20)
                plt.plot(rising_edges_t[s][i], rising_edges_mV[s][i], "*", color = "blue")
                plt.plot(rising_edges_t[s][i], slopes[s][i] * rising_edges_t[s][i] + biases[s][i], color = "red")

            for t in range(len(peak_index_min[s])):
                if plotting:
                    plt.plot(times[s], [clk_bound_min]*len(clock[s]), "--", color="gray") # Lower limit
                    plt.plot(times[s], [clk_bound_max]*len(clock[s]), "--", color="gray") # Upper limit

        clock_timestamps_10.append(t_stamps_10)
        amps_10.append(a_10)
        min_rising_edge_t.append(mins_t)
        max_rising_edge_t.append(maxes_t)
        min_rising_edge_mV.append(mins_mV)
        max_rising_edge_mV.append(maxes_mV)

        if len(clock_timestamps_10[s]) > 1:
            for p in range(1, len(clock_timestamps_10[s])):
                clock_timestamp_difference_10.append(np.abs(clock_timestamps_10[s][p] - clock_timestamps_10[s][p-1]))
                if plotting:
                    plt.plot(np.linspace(clock_timestamps_10[s][p-1], clock_timestamps_10[s][p-1] + (np.abs(clock_timestamps_10[s][p] - clock_timestamps_10[s][p-1])), 100), [amps_10[s][p]]*100, color = "black")
                    plt.plot([clock_timestamps_10[s][p-1], clock_timestamps_10[s][p-1] + (np.abs(clock_timestamps_10[s][p] - clock_timestamps_10[s][p-1]))], [amps_10[s][p-1]]*2, color = "brown", marker = "o", linestyle = "-", linewidth = 4)
                    plt.plot([clock_timestamps_10[s][p-1], clock_timestamps_10[s][p]], [amps_10[s][p-1], amps_10[s][p]], color = "black", marker = "o", linestyle = "-", linewidth = 2)
        else:
            clock_timestamp_difference_10.append(-10)
        
        if plotting:
            plt.savefig(f"./peaks_fit/plot_{s}.png")
            plt.close()
        
        clock_timestamp_differences_10.append(clock_timestamp_difference_10)
        
    return clock_timestamps_10,clock_timestamp_differences_10, n_peaks_after_truncation, n_fit_points

def merge_trees(files, trees, output_file):
    # Read ROOT files and trees
    ts = [uproot.open(files[t])[tree] for t, tree in enumerate(trees)] # if t != i else uproot.open(files[t])[tree][0::15000]]
    # Load data from trees .arrays()
    clock_old = add_clock_old(uproot.open(files[1])["pulse"])
    number_of_events = len(ts[0].arrays())
    clock_10, t_diff_10, n_peak_a, n_fit_points = add_clock(uproot.open(files[1])["pulse"], plotting = args.plotting)
    datas = [t.arrays()[0:number_of_events] for t_i, t in enumerate(ts)] # taking as many events from KCU as we have triggers

    datas.append(ak.Array({"Clock": clock_old, "Clock_1": clock_10, "t_diff": t_diff_10, "n_peaks": n_peak_a, "n_fit_points": n_fit_points}))
    
    # Merge the two datasets
    merged_data  = {}
    common_keys  = []
    other_keys_1 = []
    other_keys_2 = []

    for data in datas:
        print(data, type(data))
        for key in data.fields:
            if key not in merged_data.keys():
                merged_data[key] = data[key]

    # Create a new output file and write the merged tree
    with uproot.recreate(output_file) as output:
        for key in list(merged_data.keys()):
            # print(key)
            if "sensor" not in key:
                print(key, merged_data[key], len(merged_data[key]), type(merged_data[key]))
        output[trees[0]] = {key: merged_data[key].to_list() for key in list(merged_data.keys()) if "sensor" not in key}

if __name__ == "__main__":
    # Replace these file names and tree names with your actual file names and tree names 
    # /home/daq/ETROC2_Test_Stand/ScopeHandler/Lecroy/Conversion/merging.txt
    # /eos/uscms/store/group/cmstestbeam/2024_05_FNAL_ETL/LecroyScope/RecoData/TimingDAQRECO/RecoWithTracks/v11/run712040_info.root
    base = "/home/daq/ETROC2_Test_Stand/module_test_sw/merge_test"
    backup = "/media/daq/G-DRIVE/BACKUP_FILES"
    # os.system("./reset_merging.sh")
    f_index = args.run_number # 712085 # int(open(f"{base}/Lecroy/Acquisition/next_run_number.txt").read())
    
    print(f_index, "\n")
    prev_status = 0
    failed_iterations = 0
    iterations_limit = 1
    waiting_mode = False
    while True:
        reco_tree   = f"/eos/uscms/store/group/cmstestbeam/2024_05_FNAL_ETL/LecroyScope/RecoData/TimingDAQRECO/RecoWithTracks/v11/run{f_index}_info.root"
        etroc_tree  = f"/eos/uscms/store/group/cmstestbeam/2024_05_FNAL_ETL/ETROC_Data/root/ETROC_merged_run_{f_index}.root"
        scope_tree  = f"/eos/uscms/store/group/cmstestbeam/2024_05_FNAL_ETL/LecroyScope/RecoData/ConversionRECO/converted_run{f_index}.root"
        if not waiting_mode:
            print(reco_tree)
            print(etroc_tree)
            print(scope_tree)
        # etroc_tree  = f"{base}/RecoData/output_run_{f_index}.root" # not added yet
        merged_file = f"/eos/uscms/store/user/ahayrape/run_{f_index}.root" # f"{base}/ScopeData/LecroyMerged/run_{f_index}.root"
        reco_1 = True; # (open(f"{base}/Lecroy/Conversion/merging.txt",                    "r").read() == "True") # Conversion step (recoLoop.py)
        reco_2 = (path.isfile(reco_tree))
        reco = reco_1 and reco_2
        scope_1 = True; # (open(f"{base}/Lecroy/Acquisition/merging.txt",                  "r").read() == "True")
        scope_2 = (path.isfile(scope_tree))
        scope = scope_1 and scope_2
        etroc_1 = True; # (open(f"/home/daq/ETROC2_Test_Stand/module_test_sw/merging.txt", "r").read() == "True")
        etroc_2 = (path.isfile(etroc_tree))
        etroc = etroc_1 and etroc_2
        status = sum([reco_1, scope_1, etroc_1]) # , not path.isfile(merged_file)])

        if path.isfile(merged_file):
            print("The merged file has been already created!")
            break

        if abs(status - prev_status) > 0 and not waiting_mode:
            print( "                                  Flag      File")
            print(f"Acquisition from the scope done: {scope_1} {scope_2}")
            print(f"Acquisition from the KCU done:   {etroc_1} {etroc_2}")
            print(f"Conversion done:                 {reco_1} {reco_2}")
            print(f"Merged file wasn't created:      {not path.isfile(merged_file)}")
            print()

        # if reco and scope and etroc and (not path.isfile(merged_file)):
        if reco and scope and etroc:
            print("Secondary checking")
            print(reco)
            print(scope)
            print(etroc)
            print(not path.isfile(merged_file))
            print("\n")
            print(f"Reco data: {reco_tree}")
            print(f"Scope data: {scope_tree}")
            print(f"ETROC data: {etroc_tree}")
            print(f"Merged data: {merged_file}")
            time.sleep(5)
            merge_trees([reco_tree, scope_tree, etroc_tree], ["pulse", "pulse", "pulse"], merged_file)
            print("Merging done!")
        elif reco and scope and not etroc: # if the acquisition from the KCU has failed
            print("BREAKING")
            break
        elif not reco and scope and etroc: # if the run is not finished yet
            print(f"RECO is not finished, iteration: {failed_iterations}")
            time.sleep(10)
            failed_iterations+=1
            if failed_iterations > iterations_limit:
                failed_iterations = 0
                break
            continue
        elif not (reco and scope and etroc): # if the run didn't start yet
            print("The run didn't start yet")
            time.sleep(10)
            continue

        time.sleep(10)
        prev_status = status
        break;
