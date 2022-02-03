#!/bin/bash

for weather in 1 8 5 12 2 9
do
	python3 SafetySimulation/HMC_Mean_weather.py -f SafetySimulation/HMC_Weights -n 60 --W $weather
	path="Results/HMCout$weather"
	mv _out "$path"
	python3 SafetySimulation/HMC_Mean_weather.py -f SafetySimulation/VI_Weights -n 60 --W $weather
	path="Results/VIout$weather"
	mv _out "$path"
done

