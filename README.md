## Exploring Flexible Communications for Streamlining DNN Ensemble
   Training Pipelines ##

### Files ###





__analyze\_output.py__

Takes a directory or folder and performs averaging operations over
timing data. Designed to handle many parameters, used in accelerating
the process of data collection.

__cpu\_usage.py__

A utility file that takes a single mpstat log file as input and
converts it to a csv of non-idle resources, where first column is
time, and remaining columns are each CPU utilization.

__ensemble.py__

Main file for ensemble training.

__gen\_peak.py__

Script used to generate the data for the _peak_ function.

__gen\_util.sh__

Script that can be used to log GPU utilization out of nvidia-smi. 

__modify\_csv.py__

Utility for modifying csvs, removing columns, etc. 

__multi\_gpu\_train.py__

A script that handles multi gpu training for a single model (batch
parallelism). titan_baseline.pbs uses this script with 1 GPU for the
baseline. xsede can use this to obtain parallel training.

__parse\_mpstat.py__

Utility to parse the output of the mpstat logger (which is run within
the pipeline.py script as a simply os.system background command). 

__pipeline.py__

Abstraction for the different pipeline designs.

__summit\_train.pbs__

Train script used in initial Summit-dev tests. 

__tensorboard.sh__

IP-detector tensorboard launcher. Meant as a _sample_ to assist with
quick tensorboard launching, would require IP verification and some
modification to work on your system.

__titan\_baseline.pbs__

Run a baseline test on Titan. Script will need modification for a
specific test. 

__titan\_ensemble.pbs__

Run an ensemble technique on Titan. Script will need modification for
a specific test.

__utils.py__

Some utilities.

__xsede\_train.pbs__

Training script for xsede (caution, likely out of date). 

### Folders ###

__datasets__ | __nets__ | __preprocessing__

Slim's folders. 

__output__

Used for std output for cluster jobs.

__Logs__

Used for log files associated with the training process (other than
std out). 