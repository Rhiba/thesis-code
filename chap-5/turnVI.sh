#!/bin/bash

for unused in 0 3 11
do
	killall CarlaUE4
	./CarlaUE4.sh&
	sleep 20
	python3 SafetySimulation/Bayes.py -f SafetySimulation/VI_Weights_turn -n 100 -w $unused -s $1
	#python3 SafetySimulation/Drop.py -f SafetySimulation/DropoutTurn3.h5 -n 25 -w 0 -s A
	path="VI/Results$unused$1"	
	mkdir "$path"
	mv "run_start_"$1"_weather_"$unused "$path"
done
