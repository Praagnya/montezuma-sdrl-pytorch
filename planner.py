#!/usr/bin/env python3
# Python 3 version of the symbolic planning module with Montezuma's Revenge support

import os
import subprocess
import psutil
import signal
import shlex

actionlist = ['move']
fluentlist = ['at', 'cost', 'picked']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_result():
    with open('result.tmp') as infile:
        content = infile.readlines()
        for info in content:
            if info.startswith('Answer'):
                result = content[content.index(info)+1].strip('\n')
                return result.split(" ")
    return None

def get_type(inputstring):
    prefix = inputstring[:inputstring.find('(')]
    for act in actionlist:
        if act == prefix:
            return "action"
    for flu in fluentlist:
        if flu == prefix:
            return "fluent"


def split_time(result):
    splittedtuple = []
    for res in result:
        if "=" in res:
            equalposition = res.rfind('=')
            value = res[equalposition+1:]
            timestamp = res[5:equalposition-1]
            atom = "cost=" + value
            splittedtuple.append((int(timestamp), atom, get_type(res)))
        else:    
            index = res.rfind(',')
            timestamp = res[index+1:][:-1]
            atompart = res[:index]
            atom = "".join(atompart) + ")"
            splittedtuple.append((int(timestamp), atom, get_type(res)))
    return splittedtuple

def construct_lists(split, step):
    actions = ''
    fluents = []
    for s in split:
        if s[0] == step:
            if s[2] == 'action':
                actions = actions + s[1] + ' '
            else:
                fluents.append(s[1])
    return actions.strip(), fluents

def compute_plan(clingopath=None, initial="", goal="", planning="", qvalue="", constraint="", printout=False):
    # Add this check at the beginning to handle missing Clingo
    if clingopath is not None and not os.path.exists(clingopath):
        print(f"Warning: Clingo not found at {clingopath}")
        print("Creating mock result for Montezuma's Revenge testing")
        # Create a mock result.tmp file with Montezuma-specific states
        with open('result.tmp', 'w') as f:
            f.write("Answer: 1\n")
            f.write("move(plat1,1) at(plat1,1) move(lower_right_ladder,2) at(lower_right_ladder,2) move(devilleft,3) at(devilleft,3)\n")
    
    if printout:
        print("Generate symbolic plan...")
    if initial == "":
        initial = "initial.lp"
    if planning == "":
        planning = "taxi.lp"
    if goal == "":
        goal = "goal.lp"
    show = "show.lp"
    files = initial + " " + planning + " " + qvalue + " " + constraint + " " + goal + " " + show

    # Only try to run Clingo if the path exists
    if clingopath is not None and os.path.exists(clingopath):
        # Use subprocess.Popen with shell=True and redirect output safely
        clingconprocess = subprocess.Popen(
            f"{clingopath} {files} --time-limit=180",
            shell=True,
            cwd="/Users/praagnya/Desktop/Research/SDRL Code/sdrl/SDRL Pytorch/",
            executable="/bin/bash",
            stdout=open('result.tmp', 'w'),
            stderr=subprocess.STDOUT
        )
        p = psutil.Process(clingconprocess.pid)
        try:
            p.wait(timeout=360)
        except psutil.TimeoutExpired:
            # p.kill() is more appropriate in Python 3
            p.kill()
            print(bcolors.FAIL + "Planning timeout. Process killed." + bcolors.ENDC)
            return None
    elif clingopath is not None:
        # If we already created a mock result above, skip this
        pass
    else:
        # If no clingopath is provided and we haven't created a mock result yet
        if not os.path.exists('result.tmp'):
            print("No Clingo path provided. Creating mock result for Montezuma's Revenge.")
            with open('result.tmp', 'w') as f:
                f.write("Answer: 1\n")
                f.write("move(plat1,1) at(plat1,1) move(lower_right_ladder,2) at(lower_right_ladder,2) move(devilleft,3) at(devilleft,3)\n")
   
    result = extract_result()
    if result is None:
        return None
    split = split_time(result)
    # Get the maximum time step
    maxtime = int(sorted(split, key=lambda item: item[0], reverse=True)[0][0])
    if printout:
        print("Find a plan in", maxtime, "steps")
    plan_trace = []
    for i in range(1, maxtime + 1):
        actions, fluents = construct_lists(split, i)
        plan_trace.append((i, actions, fluents))
        if printout is True:
            print(bcolors.OKBLUE + "[TIME STAMP]", i, bcolors.ENDC)
            if fluents != '':
                print(bcolors.OKGREEN + "[FLUENTS]" + bcolors.ENDC, fluents)
            if actions != '':
                print(bcolors.OKGREEN + "[ACTIONS]" + bcolors.ENDC, bcolors.BOLD + actions + bcolors.ENDC)
    return plan_trace

# Example usage:
# compute_plan()