### if sample rate or horizontal window is changed, TimingDAQ must be recompiled to account for new npoints.
import sys
import os
sampleRate = 1000 #GSa/s ### not sure if works for Lecroy
horizontalWindow = 50 #ns
# horizontalWindow = 40 #ns
# horizontalWindow = 25 # ns
# horizontalWindow = 12.5 #ns

#Hard Code these:
trigCh="C8" #"EX" ## or "C{N}" or "EX"
trig= 1.0 ## -0.01 V

vScale1 = 0.05
vScale2 = 0.05 
vScale3 = 0.05 
vScale4 = 0.05 
vScale5 = 0.05 
vScale6 = 0.10
vScale7 = 0.50
vScale8 = 0.50
vPos6 = 0
vPos7 = 0
vPos8 = -3

trigSlope = "POS"
timeoffset = 0 #ns
# timeoffset = 12.5 #ns
runNumber = -1 ### -1 means use serial number
ScopeControlDir = "../../ScopeHandler/Lecroy/"

def ScopeAcquisition(numEvents):
	print("\n ####################### Running the scope acquisition ##################################\n")
	ScopeCommand = 'python3 %s/Acquisition/acquisition.py --runNum %s --numEvents %d --sampleRate %d --horizontalWindow %d --trigCh %s --trig %f' % (ScopeControlDir,runNumber, numEvents, sampleRate, horizontalWindow, trigCh, trig)
	ScopeCommand += ' --vScale1 %f --vScale2 %f --vScale3 %f --vScale4 %f --vScale5 %f --vScale6 %f --vScale7 %f --vScale8 %f' % (vScale1, vScale2, vScale3, vScale4, vScale5, vScale6, vScale7, vScale8)
	ScopeCommand += ' --vPos6 %f --vPos7 %f --vPos8 %f ' % (vPos6, vPos7, vPos8)
	ScopeCommand += ' --timeoffset %i --trigSlope %s --display 1' % (timeoffset,trigSlope) 
	print(ScopeCommand)
	os.system(ScopeCommand)
		    
if __name__ == "__main__":
	kcu_acquisition_flag = open("../../module_test_sw/running_ETROC_acquisition.txt").read()
	print(kcu_acquisition_flag)
	iteration = 0
	while kcu_acquisition_flag == "False":
	    if iteration == 0:
	        print(f"Waiting for the KCU.")
	    kcu_acquisition_flag = open("../../module_test_sw/running_ETROC_acquisition.txt").read()
	    iteration+=1
	f = open("../../ScopeHandler/Lecroy/Acquisition/running_acquitision.txt", "w")
	f.write("True")
	f.truncate()
	f.close()
	numEvents = int(sys.argv[1])
	ScopeAcquisition(numEvents)
	f = open("../../ScopeHandler/Lecroy/Acquisition/running_acquitision.txt", "w")
	f.write("False")
	f.truncate()
	f.close()
	f = open("../../ScopeHandler/Lecroy/Acquisition/merging.txt", "w")
	f.write("True")
	f.truncate()
	f.close()

