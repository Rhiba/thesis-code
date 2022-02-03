#!/bin/bash

#for weather in 0 1 2 3 10 11
for loc in B C D
do
	./turnHMC.sh $loc
	./turnVI.sh $loc
	./turnDrop.sh $loc
done
