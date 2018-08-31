#Searches through directory provided and averages training times for each file

import os
import sys
import re
import cpu_usage

RESET = "===============RESET DATA COLLECTION==============="

def get_avg_str(l):
    return str(round(sum(l)/len(l), 2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Expected 1 argument")
    else:
        folder = sys.argv[1]
        if os.path.isfile(folder):
            print("File provided, analyzing file")
            files = [folder]
            folder = os.path.dirname(folder)
        else:
            print("Directory provided, searching...")
            items = os.listdir(folder)
            files = [os.path.join(folder, f) for f in items if os.path.isfile(os.path.join(folder, f))]
            print("Found files:", files)
        #Sorting data strings by num cores
        num_columns = 15
        column_width = 18
        fmt_str = ("%-" + str(column_width) + "s") * num_columns
        data = []
        print(fmt_str % ("version", "model", "num gpus", "Time", "num times", "avg startup", "num cores", "utime", "log dir", "avg unix start", "avg unix end", "cpu integral", "images/sec", "Ensemble Size", "num pre"))
        for filename in files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                line_index = 0
                while line_index < len(lines):
                    times = []
                    model = "not found"
                    num_gpus = "not found"
                    num_cores = "not found"
                    utime = "not found"
                    log_dir = "not found"
                    cpu_integral = "not found"
                    version = "not found"
                    ensemble_size = "not found"
                    num_pre = "not found"
                    
                    unix_starts = []
                    startup_delays = []
                    unix_ends = []
                    im_per_sec = []
                    while line_index < len(lines):
                        line = lines[line_index]
                        if RESET in line:
                            line_index += 1
                            break
                        if "Completed training in " in line:
                            times.append(float(line.split(' ')[-2]))
                        if "Number of cores: " in line:
                            num_cores = line.split(' ')[-1].strip()
                        if "Application " in line and "utime" in line:
                            result = re.search("utime\ ~([0-9]+)s", line)
                            if result:
                                utime = result.group(1)
                            else:
                                print("Warning: utime regex specification error in file " + filename)
                        if "Model: " in line:
                            model = line.split(' ')[-1].strip()
                        if "Number of GPUs: " in line:
                            num_gpus = line.split(' ')[-1].strip()
                        if "Using log directory" in line:
                            log_dir = line.split(' ')[-1].strip()
                            cpu_integral = str(round(cpu_usage.cpu_usage_integral(log_dir), 1))
                        if "Startup time = " in line:
                            delay = float(line.split(' ')[-1].strip())
                            startup_delays.append(delay)
                        if "Version:" in line:
                            version = line.split(' ')[-1].strip()
                        if "Main started at:" in line:
                            unix_starts.append(int(line.split(' ')[-1]))
                        if "Completed training at:" in line:
                            unix_ends.append(int(line.split(' ')[-1]))
                        if "images/sec" in line:
                            im_per_sec.append(float(line.split(' ')[-1]))
                        if "ENSEMBLE SIZE:" in line:
                            ensemble_size = line.split(' ')[-1].strip()
                        if "NUM_PRE:" in line:
                            num_pre = line.split(' ')[-1].strip()
                        line_index += 1
                    avg_unix_start = "not found"
                    avg_startup_delay = "not found"
                    avg_unix_end = "not found"
                    avg_im_per_sec = "not found"
                    if len(unix_starts) != 0:
                        avg_unix_start = str(round(sum(unix_starts) / len(unix_starts), 2))
                    if len(unix_ends) != 0:
                        avg_unix_end = str(round(sum(unix_ends) / len(unix_ends), 2))
                    if len(startup_delays) != 0:
                        avg_startup_delay = str(round(sum(startup_delays) / len(startup_delays), 2))
                    if len(im_per_sec) != 0:
                        avg_im_per_sec = get_avg_str(im_per_sec)
                    # if ensemble_size != "not found":
                    #     assert(len(im_per_sec) == int(ensemble_size))
                    # print(len(unix_starts), len(unix_ends), len(startup_delays))
                    # print(fmt_str % ("model", "num gpus", "Time", "num times", "num cores", "utime"))
                    avg_time = "no times" if len(times) == 0 else str(round(sum(times)/len(times), 1))
                    num_times = str(len(times))
                    entry = (version, model, num_gpus, avg_time, num_times, avg_startup_delay, num_cores, utime, log_dir, avg_unix_start, avg_unix_end, cpu_integral, avg_im_per_sec, ensemble_size, num_pre)
                    print(fmt_str % entry)
                    data.append(entry)
        # write csv
        # if not os.path.isfile(folder):
        csv_fmt_str = ("%s," * (num_columns-1)) + "%s\n"
        with open(os.path.join(folder, 'all_data.csv'), 'w') as out:
            for entry in data:
                out.write(csv_fmt_str % entry)
                    #desc_str = "File %s has average training time %f over %d nodes, with %s cores found, with utime %d" % (filename, -1 if len(times) == 0 else sum(times)/len(times), len(times), "no" if num_cores == -1 else str(num_cores), utime)
        #         all_data.append([num_cores, desc_str])
        # all_data.sort()
        # for entry in all_data:
        #     print(entry[-1])
