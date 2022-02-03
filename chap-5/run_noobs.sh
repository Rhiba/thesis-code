#!/bin/bash

#for weather in 0 1 2 3 10 11
for weather in 0 3 11
do
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 60
	python3 SafetySimulation/HMC_Mean_obs_weather.py -f NetworkTraining/NoObstacle/HMC_Weights_NOobstacle -n 200 --W $weather
	path="Results/HMCout$weather"
	mv _out "$path"
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 60
	python3 SafetySimulation/HMC_Mean_obs_weather.py -f NetworkTraining/NoObstacle/VI_Weights_NOobstacle -n 200 --W $weather
	path="Results/VIout$weather"
	mv _out "$path"
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 60
	python3 SafetySimulation/Dropout_Mean_obs_weather.py -f NetworkTraining/NoObstacle/Dropout_discrete_NOobstacle.h5 -n 200 --W $weather
	path="Results/Dropout$weather"
	mv _out "$path"
done
