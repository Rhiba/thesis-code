#!/bin/bash

#for weather in 0 1 2 3 10 11
for location in A B C D
do
	for weather in 0 3 5
	do
		killall CarlaUE4
		./CarlaUE4.sh&
		sleep 20
		python3 SafetySimulation/HMC_Mean_obs_weather.py -f SafetySimulation/HMC_Weights_obstacle -n 5 --W $weather -l $location
		path="Results/HMCout$weather$location"
		mv _out "$path"
		killall CarlaUE4
		./CarlaUE4.sh&
		sleep 20
		python3 SafetySimulation/HMC_Mean_obs_weather.py -f SafetySimulation/VI_Weights_obstacle -n 5 --W $weather -l $location
		path="Results/VIout$weather$location"
		mv _out "$path"
		killall CarlaUE4
		./CarlaUE4.sh&
		sleep 20
		python3 SafetySimulation/Dropout_Mean_obs_weather.py -n 5 -f Dropout_discrete_obstacle.h5 --W $weather -l $location
		path="Results/Dropout$weather$location"
		mv _out "$path"
	done
done
