# def RecoAllScope(laserMode=False, LGADChannel=2, LGADThreshold=50):
import time
from datetime import datetime
# import numpy as np
# from numpy import loadtxt
import getpass
import os
import subprocess
import socket
import sys
import glob
import shutil
import ROOT
import argparse

####Use setup script in home area (setup.sh)

raw_path = "/home/daq/LecroyMount/" # ADD THE LECROY SCOPE RAW DATA FOLDER
converted_path = "../../ScopeHandler/ScopeData/LecroyConverted/"
reco_path ="../../ScopeHandler/ScopeData/LecroyTimingDAQ/"

parser = argparse.ArgumentParser(description='Run info.')

parser.add_argument('--singleMode',metavar='Single event mode', type=str,default = 0, help='Single event acquisition (default 0, 1 for yes)',required=False)
args = parser.parse_args()

useSingleEvent =False
if int(args.singleMode) > 0:
    useSingleEvent=True
    print("Using single event mode.")

def RunEntriesScope(FileLocation, LGADChannels, LGADThreshold):
    list_hits=[]
    list_coinc=[]
    f = ROOT.TFile.Open(FileLocation)
    if hasattr(f, 'pulse'):
        ##### Number of Entries with Tracks
        TotalEntries = f.pulse.GetEntries()
        for chan in LGADChannels:
            EntriesWithLGADHits = f.pulse.GetEntries("amp[%i]>%i"%(chan,LGADThreshold))
            CoincRate = float(EntriesWithLGADHits)/TotalEntries
            list_hits.append(EntriesWithLGADHits)
            list_coinc.append(CoincRate)
        return list_coinc, list_hits, TotalEntries
    else:
        return -1

LGADChannels=[0,1,2,7]
Threshold=15
acquisition_ready         = open("../../ScopeHandler/Lecroy/Acquisition/merging.txt",             "r").read() # This flag says if the acquisition data is ready to be merged.
running_acquitision_SCOPE = open("../../ScopeHandler/Lecroy/Acquisition/running_acquitision.txt", "r").read() # This flag says if the scope acquisition is still running.
running_acquitision_ETROC = open("../../module_test_sw/running_ETROC_acquisition.txt",            "r").read() # This flag says if the KCU acquisition is still running.
print("acquisition_ready: ", acquisition_ready)
print("running_acquitision_SCOPE: ", running_acquitision_SCOPE)
print("running_acquitision_ETROC: ", running_acquitision_ETROC)

while True:
    acquisition_ready         = open("../../ScopeHandler/Lecroy/Acquisition/merging.txt",             "r").read() # This flag says if the acquisition data is ready to be merged.
    running_acquitision_SCOPE = open("../../ScopeHandler/Lecroy/Acquisition/running_acquitision.txt", "r").read() # This flag says if the scope acquisition is still running.
    running_acquitision_ETROC = open("../../module_test_sw/running_ETROC_acquisition.txt",            "r").read() # This flag says if the KCU acquisition is still running.
    if (not (running_acquitision_SCOPE == "False" and acquisition_ready == "True")): continue
    ListRawFiles = [(x.split('C8--Trace')[1].split('.trc')[0]) for x in glob.glob('%s/C8--Trace*'%raw_path)]
    SetRawFiles = set([int(x) for x in ListRawFiles])
    #### reprocess hack
    #  SetRawFiles = set( range(153629,154321))#range(9500,11099) )
        #+range(153629,154321))
    #  print "Found files: "
    #  print SetRawFiles

    if len(SetRawFiles) != 0:
        print(f"Number of files to be converted: {len(SetRawFiles)}")
    # print(ListRawFiles)
    # print(glob.glob('%s/C8--Trace*'%raw_path))

    for run in SetRawFiles:
        RecoPath = '%s/converted_run%i.root' % (converted_path,run)
        RawPath = 'C5--Trace%i.trc' % run

        print('lsof -f --/home/daq/LecroyMount/%s |grep -Eoi %s' % (RawPath, RawPath))
        if os.path.exists(RecoPath):
            print('Run %i already converted. Doing reco stage two' % run)

        elif not os.popen('lsof -f -- /home/daq/LecroyMount/%s |grep -Eoi %s' % (RawPath, RawPath)).read().strip() == RawPath:
            print('Converting run ', run)
            if not useSingleEvent: 
                print("using conversion")
                ConversionCmd = "python3 ../../ScopeHandler/Lecroy/Conversion/conversion.py --runNumber %i" % (run)
            else:
                print("using one event conversion")
                ConversionCmd = "python3 ../../ScopeHandler/Lecroy/Conversion/conversion_one_event.py --runNumber %i" % (run)
            os.system(ConversionCmd)
        
        if useSingleEvent: continue
        print('Doing dattoroot for run %i' % run)
        
        OutputFile = '%s/run_scope%i.root' % (reco_path, run)
        # OutputFile = '%s/run_scope%i.root' % (converted_path, run)
        DattorootCmd = '../../TimingDAQ/NetScopeStandaloneDat2Root --correctForTimeOffsets --input_file=%s/converted_run%i.root --output_file=%s --config=/home/daq/ETROC2_Test_Stand/TimingDAQ/config/LecroyScope_v12.config --save_meas'  % (converted_path,run,OutputFile)
        # need the correct executable script to make the reco files

        print(DattorootCmd)
        os.system(DattorootCmd)
        can_be_later_merged = False
        try:
            CoincRate, EntriesWithLGADHits, TotalEntries = RunEntriesScope(OutputFile, LGADChannels, Threshold) # lgad channel starting from zero 
            can_be_later_merged = True
        except Exception as error:
            print(repr(error))
        print("Run %i: Total entries are %i"%(run,TotalEntries))
        for i,chan in enumerate(LGADChannels):
            print('\t Channel %i coincidence:  %.1f%% (%i hits)'  % (chan+1, 100.*CoincRate[i], EntriesWithLGADHits[i]))
        print("\n")

        print('Now moving the converted and raw data to backup')
        # # #Here moving the converted and raw data to backup
        os.system('mv ../../ScopeHandler/ScopeData/LecroyRaw/C*--Trace%i.trc /media/daq/ScanBackup/LecroyBackup/Raw/' % (run)) # ADD THE BACKUP FOLDER
        print(run)

        # #Here making a link from the ScopeData directory to the backup
        for i in range(1,5): #
            os.system('ln -s /media/daq/ScanBackup/LecroyBackup/Raw/C%i--Trace%i.trc /home/daq/ScopeData/LecroyRaw/' % (i,run)) # ADD THE BACKUP FOLDER
        os.system('ln -s /media/daq/ScanBackup/LecroyBackup/Converted/converted_run%i.root %s/converted_run%i.root' % (run,converted_path,run)) 
        print('Done Moving and creating the link')
        if can_be_later_merged:
            f = open("./merging.txt", "w")
            f.write("True")
            f.truncate()
            f.close()

    time.sleep(2)
