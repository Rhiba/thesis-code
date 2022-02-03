#!/bin/bash

#for weather in 0 1 2 3 10 11
for location in A
do
	for weather in 0
	do
		killall CarlaUE4
		./CarlaUE4.sh&
		sleep 20
		python3 SafetySimulation/HMC_Mean_obs_avoid.py -f SafetySimulation/VI_Weights_obstacle -n 1 --W $weather -l $location
		python3 SafetySimulation/HMC_Mean_obs_avoid.py -f SafetySimulation/VI_Weights_obstacle -n 2 --W $weather -l $location
		python3 SafetySimulation/HMC_Mean_obs_avoid.py -f SafetySimulation/VI_Weights_obstacle -n 3 --W $weather -l $location
		python3 SafetySimulation/HMC_Mean_obs_avoid.py -f SafetySimulation/VI_Weights_obstacle -n 4 --W $weather -l $location
		python3 SafetySimulation/HMC_Mean_obs_avoid.py -f SafetySimulation/VI_Weights_obstacle -n 5 --W $weather -l $location
		path="Results/VIout$weather$location"
		mv out "$path"
	done
done
