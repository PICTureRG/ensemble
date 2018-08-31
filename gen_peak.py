"""
File parser designed to generate a csv of "peak" function data. 
Takes in_file is input arg
Outputs in_file.csv containing average throughput
"""
import sys, os


# 1: Look for ENSEMBLE SIZE: n
# 2: Average n "Node: ..." messages
# 3: store (n, avg) as a data point

def parse_file(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        find_ensemble_mode = True
        num_nodes = -1
        # num_nodes = 150
        num_found = 0
        throughputs = []
        while i < len(lines):
            line = lines[i].strip()
            if find_ensemble_mode:
                if "ENSEMBLE SIZE" in line:
                # if "NUM_PRE:" in line:
                    num_nodes = int(line.split(' ')[-1])
                    find_ensemble_mode = False
            else:
                if "Node:" in line:
                    throughputs.append(float(line.split(' ')[-1]))
                    if len(throughputs) == num_nodes:
                        #Done with this entry
                        avg = round(sum(throughputs)/len(throughputs), 3)
                        data.append((num_nodes, avg))
                        #Reset
                        find_ensemble_mode = True
                        num_nodes = -1
                        num_found = 0
                        throughputs = []
            i += 1
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Expected filename")
    else:
        filename = sys.argv[1]
        if os.path.isfile(filename):
            data = parse_file(filename)
            outfile = filename + ".csv"
            with open(outfile, 'w') as out:
                for entry in data:
                    out.write("%d, %.3f\n" % entry)
            print("Wrote to output file \"%s\"" % outfile)
        else:
            print("File not found")
