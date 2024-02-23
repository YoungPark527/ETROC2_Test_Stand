next_file_index="/home/daq/ETROC2_Test_Stand/ScopeHandler/Lecroy/Acquisition/next_run_number.txt"
index=`cat $next_file_index`
echo $index
ipython3 fnal_laser_test.py -- --kcu 192.168.0.10 --hard_reset --offset $2
echo -n "True" > running_ETROC_acquisition.txt # A flag that indicates that the acquisition from the KCU is ongoing.
(python3 daq.py --l1a_rate 0 --ext_l1a --kcu 192.168.0.10 --run $index) & # Run the acquisition from the KCU.
(sleep 15
python3 ~/ETROC2_Test_Stand/ScopeHandler/Lecroy/Acquisition/acquisition_wrapper.py $1) # Run the acquisition from the scope.

python3 data_dumper.py --input $index # Convert the binary data to json.
python3 root_dumper.py --input $index # Convert the json data to root.
echo -n "False" > running_ETROC_acquisition.txt # A flag that indicates that the acquisition from the KCU has finished.
echo -n "True" > merging.txt # A flag that indicates that the data taken from the KCU are ready to be merged.
echo -n "True" > ~/ETROC2_Test_Stand/ScopeHandler/Lecroy/Acquisition/merging.txt # A flag that indicates that the data taken from the scope are ready to be merged.
