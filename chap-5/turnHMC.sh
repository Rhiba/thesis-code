#!/bin/bash

#for unused in 1 2 3 4 5 6 7 8 
#do
for weather in 0 3 11
do
		killall CarlaUE4
		./CarlaUE4.sh&
		sleep 20
		python3 SafetySimulation/BayesHMC.py -f SafetySimulation/HMC_Weights_turn -n 100 -w $weather -s $1
		#python3 SafetySimulation/Drop.py -f SafetySimulation/DropoutTurn3.h5 -n 25 -w 0 -s A
		path="HMC/Results$weather$1"	
		mkdir "$path"
		mv "run_start_"$1"_weather_"$weather "$path"
done
