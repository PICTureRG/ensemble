# AUTHOR: Randall Pittman
# Formatted for Py3

# This file takes a single mpstat log file as input and converts it 
# to a csv of non-idle resources, where first column is time, and
# remaining columns are each CPU utilization.
# Note that the time column is in seconds from the first mpstat entry
# found in the script

#The output csv can be imported directly into an excel file, or matlab
#using "csvread", or probably matplotlib

#WARNING: The script assumes the first data block is in correct syntax
#and is not corrupted!

from __future__ import print_function

import sys, os, re

start_time = ""
global_num_cpus = -1

# Returns t1 - t2 in seconds for times formmated as HH:MM:SS XM
def time_difference(t1, t2):
    h1, m1, s1 = list(map(int, t1.split(':')))
    h2, m2, s2 = list(map(int, t2.split(':')))
    hd, md, sd = [h1 - h2, m1 - m2, s1 - s2]
    return ((3600 * hd) + (60 * md) + sd) % (12*3600)

line_index = 0
#POST: Returns line, increments index, raises IndexError if line is not long enough
def fetch_line(lines):
    global line_index
    line = lines[line_index]
    line_index += 1
    # if len(line) >= 2 and len(line) < 96 and line_index != 1:
    #     print("len was", len(line), "index was", line_index)
    #     raise IndexError("Line not long enough")
    return line

def get_data(lines):
    global line_index
    #When num_cpus=-1, need to find number and set global_num_cpus. 

class Parser:
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise ValueError("Filename \"%s\" does not exist" % filename)
        self.dirname = os.path.dirname(filename)
        with open(filename, "r") as f:
            self.lines = f.readlines()
        self.index = 0
        self.num_cpus = -1
        self.start_time = ""
        self.data = []
        
    #Moves index past the all cpu line to what is theoretically the first cpu core output
    def cycle_to_block_start(self):
        if self.index == len(self.lines): return False
        while "all" not in self.lines[self.index]:
            self.index += 1
            if self.index == len(self.lines): return False
        self.index += 1
        if self.index == len(self.lines): return False
        return True
        
        
    def parse_block(self):
        count_num_cpus = 0
        entry = []
        #Parse data
        #regex = "\d\d:\d\d:\d\d\s*\d+(\d+\.\d+){9}(\d+\.\d+)"
        time_regex = "\d+:\d+:\d+"
        #This regex groups 9 floats followed by variable amounts of
        #spaces, then 1 more float. This last float is the idle time,
        #and must be followed by the end of the string. The idle time
        #can then be referenced as group 2. 
        # idle_regex = "(\d+\.\d+\s*){9}(\d+\.\d+)$"
        idle_regex = "(\d+\.\d+)(\s*\d+\.\d+){9}$"
        line = self.lines[self.index]
        time_str = re.search(time_regex, line).group(0)
        if self.start_time == "":
            self.start_time = time_str
        entry.append(time_difference(time_str, self.start_time))
        while len(line) != 0 and self.index < len(self.lines):
            line = self.lines[self.index]
            idle_regex_result = re.search(idle_regex, line)
            if idle_regex_result:
                # entry.append(100.-float(idle_regex_result.group(1)))
                entry.append(float(idle_regex_result.group(1)))
                count_num_cpus += 1
            else:
                # print("Invalid mpstat entry encountered")
                break
            self.index += 1
        #Check num cpus
        if self.num_cpus != -1 and count_num_cpus != self.num_cpus:
            # print("Note: Invalid data found, ignored")
            pass
        else:
            if self.num_cpus == -1:
                self.num_cpus = count_num_cpus
            else:
                # Then self.num_cpus == count_num_cpus, valid case
                pass
            self.data.append(entry)
    
    #Returns data and num_cpus
    def parse(self):
        while self.cycle_to_block_start():
            self.parse_block()
        return self.data, self.num_cpus

def parse_file(filename):
    parser = Parser(filename)
    return parser.parse()

def parse_dir(dir_name):
    if not os.path.isdir(dir_name):
        raise ValueError("Dir name \"%s\" is not a valid directory" % dir_name)
    files = []
    for direc in list(os.walk(dir_name)):
        if "cpu_util.log" in direc[-1]:
            files.append(direc[0] + "/cpu_util.log")
    # print("Parse dir files:", files)
    all_data = [parse_file(file)[0] for file in files]
    return all_data

def avg(l):
    return sum(l) / len(l)

def integrate_usage(node):
    #averages elements: [time, avg_core_usage]
    averages = [[block[0], avg(block[1:])] for block in node]
    # print("avg:", avg([a[1] for a in averages]))
    total = 0
    for i in range(len(averages)-1):
        t_delta = averages[i+1][0] - averages[i][0]
        avg_usage = (averages[i+1][1] + averages[i][1])/2.
        total += avg_usage * t_delta
    return total

def cpu_usage_integral(dir_name):
    if not os.path.isdir(dir_name):
        raise ValueError("Dir name \"%s\" is not a valid directory" % dir_name)
    old_result_name = os.path.join(dir_name, "___cpu_integral.cache")
    # if os.path.exists(old_result_name) and os.path.isfile(old_result_name):
    #     with open(old_result_name, 'r') as f:
    #         return float(f.readline().strip())
    
    all_data = parse_dir(dir_name)
    # print("found", len(all_data), "nodes")
    node_integrals = [integrate_usage(node) for node in all_data]
    # print(node_integrals)
    result = sum(node_integrals)
    with open(old_result_name, 'w') as f:
        f.write(str(result) + '\n')
    return result
    
            
    # global start_time, global_num_cpus, line_index
    # line_index = 0
    # start_time = ""
    # global_num_cpus = -1
    # data = []
    # with open(filename, 'r') as f:
    #     lines = f.readlines()
    #     parser = Parser(lines)
    #     while parser.cycle_to_block_start():
    #         parser.parse_block()
        
    #     while line_index < len(lines):
    #         try:
    #             line = fetch_line(lines)
    #             # if len(line) == 0 or f.eof(): break
    #             while line.count("all") == 0:
    #                 line = fetch_line(lines)
    #             time_str = line.split(' ')[0]
    #             if start_time == "": #First block
    #                 start_time = time_str
    #             time = time_difference(time_str, start_time) % (12 * 3600)
    #             data.append([time])
    #             num_cpus = 0
    #             line = fetch_line(lines)
    #             while len(line) > 0:
    #                 num_cpus += 1
    #                 idle = float(line.split(' ')[-1])
    #                 data[-1].append(100.0 - idle)
    #                 line = fetch_line(lines)
                
    #             if global_num_cpus == -1:
    #                 global_num_cpus = num_cpus
    #             elif global_num_cpus != num_cpus:
    #                 print("LOG ERROR: Found %d and %d cpus in different blocks" % (global_num_cpus, num_cpus))
    #         except IndexError:
    #             print("Warning: Trailing data encountered")
    #             line_index = len(lines)
    # if global_num_cpus != -1 and len(data[-1]) != global_num_cpus+1:
    #     # Remove incomplete data
    #     data.pop()
    # return data, num_cpus


def write_data(data, filename):
    with open(filename, "w") as f:
        for row in data:
            for i, el in enumerate(row):
                f.write(str(el))
                if i != len(row)-1:
                    f.write(",")
                else:
                    f.write("\n")

def write_csv(filename):
    print("Reading \"" + filename + "\"")
    data, num_cpus = parse_file(filename)
    print("Found %d data blocks" % len(data))
    print("Found %d cpus" % num_cpus)
    filename.split('/')[-1]
    ext_dot_idx = filename.find(".")
    #Check whether there is an ext to the filename
    if ext_dot_idx != -1:
        output_file = filename[:ext_dot_idx] + ".csv"
    else:
        output_file = filename + ".csv"
    # output_file = os.path.join(output_dir, output_file)
    print("Writing data to \"%s\"" % output_file)
    write_data(data, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Expected Log file name or directory")
        print("Example usage: python3 my_log_file.log")
    else:
        name = sys.argv[1]
        if os.path.isfile(name):
            print("Received filename")
            write_csv(name)
        else:
            files = []
            outs = []
            print("Received directory, searching for cpu_util.log files recursively...")
            for direc in list(os.walk(name)):
                if "cpu_util.log" in direc[-1]:
                    files.append(direc[0] + "/cpu_util.log")
                    print(direc[0])
                    # Find 10 ints
                    result = re.search("([0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9])", direc[0])
                    if result:
                        print("id tag for file:", result.group(1))
                        outs.append(result.group(1))
                    else:
                        print("ERROR: re search couldn't find an ending integer dirname")
                    # outs.append(
            # files = [os.path.join(name, f) for f in os.listdir(name) if os.path.isfile(os.path.join(name,f))]
            print("Found cpu logs:", files)
            output_dir = os.path.join(name, "cpu_usage")
            if not os.path.exists(output_dir):
                # print("Error: Trying to write results to '%s' failed, file or directory exists" % output_dir)
                os.mkdir(output_dir)
            for i, filename in enumerate(files):
                print("Reading \"" + filename + "\"")
                data, num_cpus = parse_file(filename)
                print("Found %d data blocks" % len(data))
                print("Found %d cpus" % num_cpus)
                out_file = os.path.join(name, outs[i] + ".csv")
                print("Writing data to \"%s\"" % out_file)
                write_data(data, out_file)
