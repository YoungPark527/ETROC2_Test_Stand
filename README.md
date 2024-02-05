# ETROC2 Test Stand software.
The test stand consists of the radiation source, the module and the readout board which are attached to a cooling system, the KCU DAQ assistant and the oscilloscope.

# Setting up python enviromnent:
'''
cd ./TimingDAQ/
source setup_cmslpc_python3.sh
'''


1) Mount the Lecroy data from the scope
'''
cd ETROC2_Test_Stand
source MountLecroyData.sh 
'''

2) Do the acquisition
'''
cd ETROC2_Test_Stand
source Acquisition.sh
python3 acquisition_wrapper.py $number_of_events
'''

3) Conversion
'''
cd ETROC2_Test_Stand
source Conversion.sh
pytohn3 recoLoop.py
'''

5) Merging
'''
cd ETROC2_Test_Stand
source Merging.sh
python3 merge_scope_etroc.py
'''

6) ETROC data taking
'''
cd /home/daq/
source Test_Stand.sh
ipython test_tamalero.py -- --control_hub --kcu 192.168.0.10 --verbose --configuration modulev0b --power_up
ipython -i test_ETROC.py -- --test_chip --hard_reset --partial --configuration modulev0b --module 1 
(and make sure that the lpGBT connections are locked)
./reset_and_run.sh $number_of_events
'''

To run the Acquisition and the ETROC data taking and Autopilot can be used:

Autopilot
'''
cd /home/daq/
source Autopilot.sh
./autopilot.sh $number_of_events $threshold_offset
'''
