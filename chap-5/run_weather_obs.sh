#!/bin/bash

#for weather in 0 1 2 3 10 11
for weather in 3 5 10 11
do
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 60
	python3 SafetySimulation/HMC_Mean_obs_weather.py -f SafetySimulation/HMC_Weights_obstacle -n 10 --W $weather
	path="Results/HMCout$weather"
	mv _out "$path"
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 60
	python3 SafetySimulation/HMC_Mean_obs_weather.py -f SafetySimulation/VI_Weights_obstacle -n 10 --W $weather
	path="Results/VIout$weather"
	mv _out "$path"
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 60
	python3 SafetySimulation/Dropout_Mean_obs_weather.py -n 10 --W $weather
	path="Results/Dropout$weather"
	mv _out "$path"
done
