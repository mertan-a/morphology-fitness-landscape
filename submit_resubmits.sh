#!/bin/sh

while :
do
    # check the folder for files starting with resubmit
    for i in *
    do
        #echo $i
        if [[ "$i" = resubmit*.sh ]]
        then
            sbatch $i
            rm $i
        fi
    done
    sleep 300
done

