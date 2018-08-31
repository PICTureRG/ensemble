# AUTHOR: Randall Pittman
# Formatted for Py3

# This file takes a single mpstat log file as input and converts it
# to a csv of non-idle resources, where first column is time, and
# remaining columns are each CPU utilization.
# Note that the time column is in seconds from the first mpstat entry
# found in the script

#The output csv can be imported directly into an excel file, or matlab
#using "csvread", or probably matplotlib

import sys

start_time = ""
global_num_cpus = -1

# Returns t1 - t2 in seconds for times formmated as HH:MM:SS
def time_difference(t1, t2):
    h1, m1, s1 = list(map(int, t1.split(':')))
    h2, m2, s2 = list(map(int, t2.split(':')))
    hd, md, sd = [h1 - h2, m1 - m2, s1 - s2]
    return (3600 * hd) + (60 * md) + sd

def parse_file(filename):
    global start_time, global_num_cpus
    data = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline().strip()
            if len(line) == 0: break
            while line.count("all") == 0:
                line = f.readline().strip()
            if start_time == "": #First block
                start_time = line.split(' ')[0]
            time_str = line.split(' ')[0]
            time = time_difference(time_str, start_time)
            data.append([time])
            num_cpus = 0
            line = f.readline().strip()
            while len(line) > 0:
                num_cpus += 1
                idle = float(line.split(' ')[-1])
                data[-1].append(100.0 - idle)
                line = f.readline().strip()
            if global_num_cpus == -1:
                global_num_cpus = num_cpus
            elif global_num_cpus != num_cpus:
                print("LOG ERROR: Found %d and %d cpus in different blocks" % (global_num_cpus, num_cpus))

    return data, num_cpus


def write_data(data, filename):
    with open(filename, "w") as f:
        for row in data:
            for i, el in enumerate(row):
                f.write(str(el))
                if i != len(row)-1:
                    f.write(",")
                else:
                    f.write("\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Expected Log file name")
        print("Example usage: python3 my_log_file.log")
    else:
        filename = sys.argv[1]
        print("Reading \"" + filename + "\"")
        data, num_cpus = parse_file(filename)
        print("Found %d data blocks" % len(data))
        print("Found %d cpus" % num_cpus)

        ext_dot_idx = filename.find(".")
        #Check whether there is an ext to the filename
        if ext_dot_idx != -1:
            output_file = filename[:ext_dot_idx] + ".csv"
        else:
            output_file = filename + ".csv"
        print("Writing data to \"%s\"" % output_file)
        write_data(data, output_file)
