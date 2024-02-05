import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
from scipy.optimize import curve_fit
import os
import sys
import pdb
import pandas as pd


base             = "/home/daq/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged"
# legend_label     = "Laser Width"
# scanning_version = "TEST_Laser_width"
legend_label     = "TEST_Second_peak"
scanning_version = "TEST_Second_peak"
output_folder    = f"/home/daq/ETROC2_Test_Stand/module_test_sw/analysis/plots_{scanning_version}"
unit             = "V"
os.system(f"mkdir -p {output_folder}")

# y_axis = [10898,10899, 10900]
# x_axis = [185, 195, 205]

y_axis = [10904]
x_axis = [205]

a = 10
c = 1
do_Delta_T_Gaussian_fit = True

def merge_unique(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    if len(merged_list) != 0:
        merged_list = len(set(merged_list))
    else:
        merged_list = len([])
    return merged_list

def get_unique(row, col):
    pairs = ak.Array([merge_unique(row[i], col[i]) for i in range(len(row))])
    return np.array(pairs)

def load_data(indices):
    output = []
    for i, index in enumerate(indices):
        filename = f"{base}/run_{index}.root"
        output.append({})
        with uproot.open(filename) as f:
            tree = f["pulse"]
            output[i]["nhits"] = tree["nhits"].array()
            tbin               = 3.125 / tree["cal_code"].array()
            output[i]["TOA"]   = 12.5 - tbin * tree["toa_code"].array()
            output[i]["TOT"]   = (2 * tree["tot_code"].array() - np.floor(tree["tot_code"].array() / 32)) * tbin
            output[i]["CAL"]   = tree["cal_code"].array()
            output[i]["row"]   = tree["row"].array()
            output[i]["col"]   = tree["col"].array()
            # Get the timestamps
            for c in [6, 7]: # The index of the scope channel
                for p in [20]: # The percentage of amplitude value
                    if c == 6:
                        output[i][f"LP2_{p}_{c}"] = tree["Clock"].array() # [:, c] # tree[f"linear_RE_{p}"].array()[:, c]
                    if c == 7:
                        output[i][f"LP2_{p}_{c}"] = tree[f"LP2_{p}"].array()[:, c]
    return output

def gaussian(x, a, mean, sigma):
    return (a/np.sqrt(2 * np.pi * sigma)) * np.exp(-(x - mean)**2 / (2 * sigma**2))

def chi_squared(obs, pred):
    diff = [((obs[i] - pred[i])**2 / pred[i]) for i in range(len(pred))]
    print(sum(diff))
    print(sum(diff)/len(diff))
    return np.sum(diff)

def fit_curve(data, function, hist):
    data = np.array(data)
    mean = np.mean(data)
    rms = np.std(data)
    x = (hist[1] + (hist[1][1] - hist[1][0]) / 2)[:-1]
    return curve_fit(function, x, hist[0], p0 = [0.01, mean, rms], absolute_sigma = True)

def remove_empty(data):
    map = ak.count(data, axis = -1) > 0
    data = data[map]
    return data

def rescale(elem):
    if elem < 0:
        elem = elem - 25 * np.floor(elem / 25)
    elif elem > 25:
        elem = elem - 25 * np.ceil(elem / 25)
    return elem

def scan(data): # a dictionary
    print(list(data.keys()))
    print(len(data[list(data.keys())[0]]), len(data[list(data.keys())[1]]), len(data[list(data.keys())[2]]))
    maximum = max([len(data[list(data.keys())[0]]), len(data[list(data.keys())[1]]), len(data[list(data.keys())[2]])])
    if not (len(data[list(data.keys())[0]]) == len(data[list(data.keys())[1]]) == len(data[list(data.keys())[2]])):
        for i in range(maximum):
            for j in data.keys():
                print(data[j][i], end = ",")
            print()

def fill_empty_arrays(array):
    return ak.where(ak.num(array) == 0, ak.Array([[-1]]), array)

def plots(data, x_axis):
    mean_toa = [[] for i in range(4)]
    mean_tot = [[] for i in range(4)]
    mean_cal = [[] for i in range(4)]
    std_toa  = [[] for i in range(4)]
    std_tot  = [[] for i in range(4)]
    std_cal  = [[] for i in range(4)]
    std_clock   = [[] for i in range(4)]
    std_trigger = [[] for i in range(4)]
    mean_DT  = [[] for i in range(4)]
    std_DT   = [[] for i in range(4)]
    TOA      = [[] for i in range(4)]
    TOT      = [[] for i in range(4)]
    CAL      = [[] for i in range(4)]
    DT       = [[] for i in range(4)]
    hits     = []
    hits_    = [[] for i in range(len(x_axis))]
    LP2_20_6_= [[] for i in range(len(x_axis))] # Clock time stamp
    LP2_20_7_= [[] for i in range(len(x_axis))] # Trigger time stamp
    for d_i, d in enumerate(data):
        cutflow_tables = {}
        LP2_20_6 = d["LP2_20_6"]
        LP2_20_7 = d["LP2_20_7"]*10**9

        sel = ((d["nhits"] > 0)) # Select events with at least one hit.
        row = d["row"][sel]
        col = d["col"][sel]
        toa = d["TOA"][sel]
        tot = d["TOT"][sel]
        cal = d["CAL"][sel]
        LP2_20_6 = LP2_20_6[sel]
        LP2_20_7 = LP2_20_7[sel]

        hits.append(d["nhits"])
        for p in [0,1,2,3]:
            cuts = {}
            # SELECTION OF THE PIXEL
            pixel_sel = ((row == 15) & (col == p)) # Select hits from a particular pixel.
            print(f"Row: 15, Col: {p}")

            toa_ = toa[pixel_sel]
            tot_ = tot[pixel_sel]
            cal_ = cal[pixel_sel]

            obj = {"TOA": toa_, "TOT": tot_, "CAL": cal_}
            # ELEMENT LEVEL SELECTION
            element_sel = ((cal_ > 195) & (cal_ < 205) & (tot_ > 4)) # Other object level selections.
            toa_ = toa_[element_sel]
            tot_ = tot_[element_sel]
            cal_ = cal_[element_sel]

            # EVENT LEVEL SELECTION
            event_sel = ((ak.count(toa_, axis = 1) > 0) & (LP2_20_6 > -4)) # Select events for which the clock timestamp is not 0.
            toa_ = toa_[event_sel][:,0]
            tot_ = tot_[event_sel][:,0]
            cal_ = cal_[event_sel][:,0]

            LP2_20_6_[d_i] = LP2_20_6 [event_sel]
            LP2_20_7_[d_i] = LP2_20_7 [event_sel]
            hits_    [d_i] = hits[d_i][event_sel]

            mean_toa   [p].append(np.mean(toa_))
            mean_tot   [p].append(np.mean(tot_))
            mean_cal   [p].append(np.mean(cal_))
            TOA        [p].append(toa_)
            TOT        [p].append(tot_)
            CAL        [p].append(cal_)
            std_toa    [p].append(np.std (toa_))
            std_tot    [p].append(np.std (tot_))
            std_cal    [p].append(np.std (cal_))
            std_clock  [p].append(np.std (LP2_20_6_[d_i]))
            std_trigger[p].append(np.std (LP2_20_7_[d_i]))
            DT         [p].append(toa_ - (LP2_20_7_[d_i] - LP2_20_6_[d_i]))
            mean_DT    [p].append(np.mean(DT[p][d_i]))
            std_DT     [p].append(np.std (DT[p][d_i]))
            cutflow_tables[f"row_15_col_{p}"] = cuts

    # Plot graphs.
    variables = [mean_toa, mean_tot, mean_cal, std_toa, std_tot, std_cal, std_clock, std_trigger, mean_DT, std_DT]
    variables = [np.nan_to_num(var) for var in variables]
    labels    = ["mean_toa", "mean_tot", "mean_cal", "std_toa", "std_tot", "std_cal", "std_clock", "std_trigger", "mean_DT", "std_DT"]
    vars      = dict(zip(labels, variables))
    for v, var in enumerate(variables):
        plt.style.use(hep.style.CMS)
        fig, ax = plt.subplots(1, 1, figsize = (1*a*c, 1*a))
        hep.cms.label(llabel="ETL Preliminary", rlabel="")
        for p in range(4):
            var = np.array(var)
            var[var==np.inf] = 0
            if "mean" in labels[v]:
                ax.errorbar(x_axis, var[p], yerr = vars["std" + labels[v].split("mean")[1]][p], label = f"Row: 15, Col: {p}", fmt = "o", capsize = 50)
            else:
                ax.scatter(x_axis, var[p], label = f"Row: 15, Col: {p}", s=100)
            if "std_DT" == labels[v]:
                print("Row: 15, Col: ",p ,var[p])
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
        ax.set_xlabel(f"{legend_label} [{unit}]", loc='center')
        ax.set_ylabel(labels[v],loc='center')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{output_folder}/{labels[v]}.png")
        plt.close()

    # Plot histograms.
    variables = [TOA  ,  TOT ,  CAL , \
                 LP2_20_6_ ,   \
                 LP2_20_7_ ,  hits_ ,   DT]

    labels    = ["TOA", "TOT", "CAL", \
                 "LP2_20_6_", \
                 "LP2_20_7_", "hits", "DT"]

    vars      = dict(zip(labels, variables))

    for v, var in enumerate(variables):
        # t = ROOT.TText(0.5,0.0,f"Number of events: {len(var)}");
        if labels[v] == "TOA" or labels[v] == "TOT" or labels[v] == "CAL" or labels[v] == "DT":
            fig, ax = plt.subplots(2, 2, figsize = (2*a*c, 2*a))
            for p in range(4):
                minimum = min([min(var[p][i]) if len(var[p][i]) != 0 else 10000 for i in range(len(x_axis))])
                maximum = max([max(var[p][i]) if len(var[p][i]) != 0 else 0     for i in range(len(x_axis))])
                x = int(np.floor(p / 2))
                y = int(p % 2)
                try:
                    for j in range(len(x_axis)):
                        bins = np.linspace(minimum, maximum, 70)
                        weights = np.ones(len(var[p][j])) / len(var[p][j])
                        ax[x][y].hist(var[p][j], label = f"{legend_label}: {x_axis[j]} {unit}", bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                    ax[x][y].set_xlabel(labels[v],loc='center')
                    ax[x][y].set_ylabel("Events",loc='center')
                    ax[x][y].set_title(f"Row: 15, Col: {p}")
                    ax[x][y].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
                except:
                     print("No statistics.")
                handles, label = ax[x][y].get_legend_handles_labels()
                idx = np.sort(np.unique(np.array(label), return_index=True)[1])
                ax[x][y].legend(np.array(label)[idx])
                # ax[x][y].text(0.05, 0.95 - 0.05*, f"Number of events: {len(var)}")
            fig.savefig(f"{output_folder}/{labels[v]}_HIST.png")
            plt.close()
        else:
            plt.style.use(hep.style.CMS)
            fig, ax = plt.subplots(1, 1, figsize = (1*a*c, 1*a))
            hep.cms.label(llabel="ETL Preliminary", rlabel="")
            minimum = min([min(var[i]) if len(var[i]) != 0 else 10000 for i in range(len(x_axis))])
            maximum = max([max(var[i]) if len(var[i]) != 0 else 0     for i in range(len(x_axis))])
            for j in range(len(x_axis)):
                bins = np.linspace(minimum, maximum, 70)
                weights = np.ones(len(var[j])) / len(var[j])
                try:
                    if labels[v] == 'hits':
                        bins = 5
                        ran = (0,5)
                        ax.hist(var[j], label = f"{legend_label}: {x_axis[j]} {unit}",range=ran, bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                        ax.set_xlim(0,5)
                    else:
                        ax.hist(var[j], label = f"{legend_label}: {x_axis[j]} {unit}", bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                except:
                    print("No statistics.")
            ax.set_xlabel(labels[v],loc='center')
            ax.set_ylabel("Events",loc='center')
            handles, label = ax.get_legend_handles_labels()
            idx = np.sort(np.unique(np.array(label), return_index=True)[1])
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            ax.legend(np.array(label)[idx])
            fig.savefig(f"{output_folder}/{labels[v]}_HIST.png")
            plt.tight_layout()
            plt.close()
    print(len(DT))
    for i in range(len(x_axis)): # toa vs tot
        fig, ax = plt.subplots(2, 2, figsize = (2*a*c, 2*a))
        for p in range(4):
            x = int(np.floor(p / 2))
            y = int(p % 2)
            ax[x][y].hist2d(np.array(variables[0][p][i]), np.array(variables[1][p][i]), bins = (50, 50))
            ax[x][y].set_title(f"Row: 15, Col: {p}")
            ax[x][y].set_xlabel("TOA",loc='center')
            ax[x][y].set_ylabel("TOT",loc='center')
            ax[x][y].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
        fig.savefig(f"{output_folder}/TOA_vs_TOT_row_15_col_{p}_{x_axis[i]}.png")

    '''
    for i in range(len(x_axis)): # tot vs Clock
        fig, ax = plt.subplots(1, 1, figsize = (2*a*c, 2*a))
        ax.hist2d(np.array(variables[3][i]), np.array(variables[1][i]), bins = (50, 50))
        ax.set_title(f"Row: 15, Col: {p}")
        ax.set_xlabel("TOT",loc='center')
        ax.set_ylabel("Clock",loc='center')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
        fig.savefig(f"{output_folder}/TOT_Clock_{x_axis[i]}.png")

    for i in range(len(x_axis)): # DT vs Clock
        fig, ax = plt.subplots(1, 1, figsize = (2*a*c, 2*a))
        ax.hist2d(np.array(variables[6][i]), np.array(variables[1][i]), bins = (50, 50))
        ax.set_title(f"Row: 15, Col: {p}")
        ax.set_xlabel("DT",loc='center')
        ax.set_ylabel("Clock",loc='center')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
        fig.savefig(f"{output_folder}/DT_Clock_{x_axis[i]}.png")
    '''

    if do_Delta_T_Gaussian_fit:
        fig, ax = plt.subplots(2, 2, figsize = (2*a*c, 2*a))
        for p in range(4):
            x = int(np.floor(p / 2))
            y = int(p % 2)
            for j in range(len(x_axis)):
                mean_val = np.mean(DT[p][j])
                std_times_5 = 0.15 # * np.std(DT[p][j])
                minimum = mean_val - std_times_5
                maximum = mean_val + std_times_5
                if j != 3: continue
                bins = np.linspace(minimum, maximum, 31)
                weights = np.ones(len(DT[p][j])) / len(var[p][j])
                hist = ax[x][y].hist(DT[p][j], label = f"{legend_label}: {x_axis[j]} {unit}", bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                curve = fit_curve(DT[p][j], gaussian, hist)
                mean = np.mean(DT[p][j])
                x_range = np.array((hist[1] + (hist[1][1] - hist[1][0]) / 2))
                y_range = hist[0]
                x_range = x_range[0:len(y_range)]
                delta = np.abs(x_range - mean)
                sigma = np.std(delta)
                x_range = x_range[delta < 1.5*sigma]
                y_range = y_range[delta < 1.5*sigma]
                chi = np.round(chi_squared(y_range, gaussian(x_range, *curve[0])) / len(y_range), 2)
                ax[x][y].plot(x_range, gaussian(x_range, *curve[0]), "r", linewidth=2, label = "Gaussian fit\n$\sigma$ = "+str(round(curve[0][2]*1000,2))+" ps"+"\n $\chi^{2}$/N = "+str(chi)+".")
            ax[x][y].set_xlabel("$\Delta$T (ns)",loc='center')
            ax[x][y].set_ylabel("Events",loc='center')
            ax[x][y].set_title(f"Row: 15, Col: {p}")
            ax[x][y].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            handles, label = ax[x][y].get_legend_handles_labels()
            idx = np.sort(np.unique(np.array(label), return_index=True)[1])
            ax[x][y].legend(np.array(label)[idx])
        fig.savefig(f"{output_folder}/{labels[v]}_HIST_FIT.png")
        plt.close()

if __name__ == "__main__":
    data = load_data(y_axis)
    plots(data, x_axis)
