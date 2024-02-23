# ETROC2 Test Stand software.
The test stand consists of the radiation source, the module and the readout board which are attached to a cooling system, the KCU DAQ assistant and the oscilloscope (Fig.1).

<img src="./diagrams/the_setup.png" width=500></img> 

1) Mount the Lecroy data from the scope.
```
cd ETROC2_Test_Stand
source MountLecroyData.sh 
```
2) Taking data from KCU.
```
cd /home/daq/
source Test_Stand.sh
ipython test_tamalero.py -- --control_hub --kcu $DAQ_assistant_IP --verbose --configuration modulev0b --power_up
ipython -i test_ETROC.py -- --test_chip --hard_reset --partial --configuration modulev0b --module 1 
(and make sure that the lpGBT connections are locked)
./reset_and_run.sh $number_of_events
```

To run the Acquisition and the ETROC data taking at the same time the Autopilot can be used:

3) Running the autopilot.
```
cd /home/daq/
source Autopilot.sh
ipython test_tamalero.py -- --control_hub --kcu $DAQ_assistant_IP --verbose --configuration modulev0b --power_up
ipython -i test_ETROC.py -- --test_chip --hard_reset --partial --configuration modulev0b --module 1 
(and make sure that the lpGBT connections are locked)
./autopilot.sh $number_of_events $threshold_offset
```

4) Do the data acquisition from the scope.
```
cd ETROC2_Test_Stand
source Acquisition.sh
python3 acquisition_wrapper.py $number_of_events
```

5) Do the conversion of the data from the scope.
```
cd ETROC2_Test_Stand
source Conversion.sh
pytohn3 recoLoop.py
```

6) Merging the data from the scope and from the KCU (Fig.2).
```
cd ETROC2_Test_Stand
source Merging.sh
python3 merge_scope_etroc.py
```

<img src="./diagrams/data_flow.png" width=500></img>
