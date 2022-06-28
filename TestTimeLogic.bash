#!/bin/bash

now=$(date +"%T")
today=$(date +"%Y_%m_%d")
yesterday=$(date -d "yesterday" +"%Y_%m_%d")
default_pause_time=11:59:00
if [ -z $stoptime ]; then
    if [[ "$now" < "$default_pause_time" ]] ; then
      stopDate=$(date +"%Y:%m:%d")
    else 
      stopDate=$(date -d "tomorrow" +"%Y:%m:%d")
    fi 
    stoptime="$stopDate":"$default_pause_time"
fi

echo $stoptime
