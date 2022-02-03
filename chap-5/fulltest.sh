for w in 0 1 2 8
do
	for s in D
	do
		python3 SafetySimulation/HMC_Mean_weather_startpos.py -f SafetySimulation/HMC_Weights/ --W $w --N 10 -s $s
	done
done
