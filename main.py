# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# 
# Simple testing of trained subgoal models - PyTorch Implementation
# ===================================================================================================================

import os

os.environ['PYTHONHASHSEED'] = '0'

import argparse
import sys
import time
import numpy as np
import torch
from collections import namedtuple, deque
from environment_atari import ALEEnvironment
from hybrid_rl_il_agent_atari import Agent
from hybrid_model_atari import Hdqn
from simple_net import Net
from PIL import Image
from tensorboard import TensorboardVisualizer
from os import path
import time
import planner as planner 
import math, random
from hyperparameters_new import *
import pickle


def process_state_tensor(state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(state).float().to(device)
    if len(tensor.shape) == 5:
        tensor = tensor.squeeze(1)
        tensor = tensor.permute(0, 2, 1, 3)
    elif len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def pause():
    os.system('read -s -n 1 -p "Press any key to continue...\n"')

def generateplan():
    # Set clingopath explicitly
    clingopath = "/opt/homebrew/bin/clingo"
    initial = "/Users/praagnya/Desktop/Research/SDRL Code/sdrl/SDRL Pytorch/initial.lp"
    goal = "/Users/praagnya/Desktop/Research/SDRL Code/sdrl/SDRL Pytorch/goal.lp"
    planning = "/Users/praagnya/Desktop/Research/SDRL Code/sdrl/SDRL Pytorch/montezuma_basic.lp"
    qvalue = "/Users/praagnya/Desktop/Research/SDRL Code/sdrl/SDRL Pytorch/q.lp"
    constraint = "constraint.lp"
    return planner.compute_plan(
        clingopath=clingopath,
        #initial=f'"{initial}"',
        goal=f'"{goal}"',
        planning=f'"{planning}"',
        qvalue=f'"{qvalue}"',
        constraint=f'"{constraint}"',
        printout=True
    )

def calculateplanquality(ro_table, stateaction):
    planquality = 0
    for (state, action) in stateaction:
        planquality += int(math.floor(ro_table[state, action]))
    return planquality

def generate_rovalue_from_table(env, ro_table_lp, ro_table):
#    print "output qvalues"
    qfile = open("q.lp", "w")
    qfile.write("#program step(k).\n")
    for (state, action) in ro_table_lp:
        logical_state = stateRemapping(state)
        logical_action = actionRemapping(action)
        qrule = "ro("+logical_state+","+logical_action+","+str(int(math.floor(ro_table[state, action])))+").\n"
        qfile.write(qrule)
    qfile.close()

def generate_goal_file(planquality):
#    print "output new goal file"
    goalfile = open("goal.lp", "w")
    goalfile.write("#program check(k).\n")
#    goalfile.write(":- not at(key,k), query(k).\n")
    goalfile.write(":- query(k), cost(C,k), C <= "+str(planquality)+".\n")
    goalfile.write(":- query(k), cost(0,k).")
    goalfile.close()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def cleanupconstraint():
    # Only clear constraint.lp once at the start of training, not every episode
    open('constraint.lp', 'w').close()

def updateconstraint(state_ind, action_ind):
    # Appends ":-state,action." with #program step(k).
    state = stateRemappingWithTimeStamps(state_ind)
    action = actionRemappingWithTimeStamps(action_ind)
    with open("constraint.lp", "a") as f:
        f.write("#program step(k).\n")
        f.write(":-"+state+","+action+".\n")

def selectSubGoal(plantrace, i): 
    currentunit = plantrace[i]
    currentfluent = currentunit[2]
    nextunit = plantrace[i+1]
    nextfluent = nextunit[2]
    # subgoal networks, mapped from every possible symbolic transition
    # currently we only train for good ones. Will add useless(difficult) ones later.
    # make sure the goal number here maps correctly to bounding boxes in environment_atari.py

    if ("at(plat1)" in currentfluent) and ("at(lower_right_ladder)" in nextfluent) and ("picked(key)" not in nextfluent):
        return 0
    if ("at(lower_right_ladder)" in currentfluent) and ("at(devilleft)" in nextfluent):
        return 1
    if ("at(devilleft)" in currentfluent) and ("at(key)" in nextfluent):
        return 2
    if ("at(key)" in currentfluent) and ("at(lower_left_ladder)" in nextfluent):
        return 3
    if ("at(lower_left_ladder)" in currentfluent) and ("at(lower_right_ladder)" in nextfluent):
        return 4
    if ("at(lower_right_ladder)" in currentfluent) and ("at(plat1)" in nextfluent):
        return 5
    if ("at(plat1)" in currentfluent) and ("at(right_door)" in nextfluent):
        return 6
    # Add the missing transition case
    if ("at(plat1)" in currentfluent) and ("at(devilleft)" in nextfluent):
        return 7
    return -1

def obtainStateAction(plantrace, i):
    unit = plantrace[i]
    action = unit[1]
    fluent = unit[2]
    return stateMapping(fluent), actionMapping(action)

def actionMapping(action):
    if 'move(lower_right_ladder)' in action:
        return 0
    if 'move(lower_left_ladder)' in action:
        return 1
    if 'move(key)' in action:
        return 2
    if 'move(right_door)' in action:
        return 3
    if 'move(devilleft)' in action:
        return 4
    if 'move(plat1)' in action:
        return 5

def stateMapping(fluent): # symbolic state to goal mapping
    if ("at(lower_right_ladder)" in fluent) and ("picked(key)" not in fluent):
        return 0
    if ("at(key)" in fluent) and ("picked(key)" in fluent):
        return 1
    if ("at(lower_right_ladder)" in fluent) and ("picked(key)" in fluent):
        return 2
    if ("at(right_door)" in fluent) and ("picked(key)" in fluent):
        return 3
    if ("at(right_door)" in fluent) and ("picked(key)" not in fluent):
        return 4
    if ("at(devilleft)" in fluent):
        return 5
    if ("at(plat1)" in fluent) and ("picked(key)" in fluent):
        return 6
    if ("at(lower_left_ladder)" in fluent) and ("picked(key)" in fluent):
        return 7
    if ("at(lower_left_ladder)" in fluent) and ("picked(key)" not in fluent):
        return 8
    return -1

def actionRemapping(action_ind):
    if action_ind == 0:
        return 'move(lower_right_ladder)'
    if action_ind == 1:
        return 'move(lower_left_ladder)'
    if action_ind == 2:
        return 'move(key)'
    if action_ind == 3:
        return 'move(right_door)'
    if action_ind == 4:
        return 'move(devilleft)'
    if action_ind == 5:
        return 'move(plat1)'
    return ''

def stateRemapping(fluent_ind): # symbolic state to goal mapping
    if fluent_ind == -1:
        return 'at(plat1)'
    if fluent_ind == 0:
        return 'at(lower_right_ladder)'
    elif fluent_ind == 1:
        return '(at(key),picked(key))'
    elif fluent_ind == 2:
        return '(at(lower_right_ladder),picked(key))'
    elif fluent_ind == 3:
        return '(at(right_door),picked(key))'
    elif fluent_ind == 4:
        return 'at(right_door)'
    elif fluent_ind == 5:
        return 'at(devilleft)'
    elif fluent_ind == 6:
        return '(at(plat1),picked(key))'
    elif fluent_ind == 7:
        return '(at(lower_left_ladder),picked(key))'
    elif fluent_ind == 8:
        return 'at(lower_left_ladder)'
    return ''

def actionRemappingWithTimeStamps(action_ind):
    if action_ind == 0:
        return 'move(lower_right_ladder,k)'
    if action_ind == 1:
        return 'move(lower_left_ladder,k)'
    if action_ind == 2:
        return 'move(key,k)'
    if action_ind == 3:
        return 'move(right_door,k)'
    if action_ind == 4:
        return 'move(devilleft,k)'
    if action_ind == 5:
        return 'move(plat1,k)'
    return ''

def stateRemappingWithTimeStamps(fluent_ind): # symbolic state to goal mapping
    if fluent_ind == -1:
        return 'at(plat1,k)'
    if fluent_ind == 0:
        return 'at(lower_right_ladder,k)'
    elif fluent_ind == 1:
        return 'at(key,k),picked(key,k)'
    elif fluent_ind == 2:
        return 'at(lower_right_ladder,k),picked(key,k)'
    elif fluent_ind == 3:
        return 'at(right_door,k),picked(key,k)'
    elif fluent_ind == 4:
        return 'at(right_door,k)'
    elif fluent_ind == 5:
        return 'at(devilleft,k)'
    elif fluent_ind == 6:
        return 'at(plat1,k),picked(key,k)'
    elif fluent_ind == 7:
        return 'at(lower_left_ladder,k),picked(key,k)'
    elif fluent_ind == 8:
        return 'at(lower_left_ladder,k)'
    return ''

def throwdice(threshold):
    rand = random.uniform(0, 1)
    if rand < threshold:
        return True
    else:
        return False

def obtainedKey(previoustate, nextstate):
    if ("picked(key)" not in previoustate) and ("picked(key)" in nextstate):
        return True
    else:
        return False

def openDoor(previoustate, nextstate):
    if ("picked(key)" in previoustate) and ("at(right_door)" not in previoustate) and ("picked(key)" in nextstate) and ("at(right_door)" in nextstate):
        return True
    else:
        return False

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visualizer = TensorboardVisualizer()
    logdir = path.join(recordFolder+'/')  # subject to change
    visualizer.initialize(logdir, None)

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]

    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']

    # Updated goalExplain list with the new subgoal
    goalExplain = ['lower right ladder', 'jump to the left of devil', 'key', 'lower left ladder',
                   'lower right ladder', 'central high platform', 'right door', 'platform to devil']

    Num_subgoal = len(goalExplain)
    subgoal_success_tracker = [[] for i in range(Num_subgoal)]
    subgoal_trailing_performance = [0] * Num_subgoal
    random_experience = [deque() for _ in range(Num_subgoal)]
    kickoff_lowlevel_training = [False] * Num_subgoal

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=False)
    parser.add_argument("--frame_skip", default=4)
    parser.add_argument("--color_averaging", default=True)
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    parser.add_argument("--load_weight", default=False)
    parser.add_argument("--use_sparse_reward", type=str2bool, default=True)
    args = parser.parse_args()

    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])

    env = ALEEnvironment(args.game, args)

    # Set GPU to False for CPU usage
    GPU = False

    hdqn_list = [Hdqn(GPU) for _ in range(Num_subgoal)]
    agent_list = [
        Agent(hdqn, list(range(nb_Action)), list(range(Num_subgoal)),
            defaultNSample=BATCH, defaultRandomPlaySteps=20000,
            controllerMemCap=EXP_MEMORY, explorationSteps=300000,
            trainFreq=TRAIN_FREQ, hard_update=HARD_UPDATE_FREQUENCY)
        for hdqn in hdqn_list
    ]
    print("Exploration steps set to 300000 for all agents.")

    for i in range(Num_subgoal):
        agent_list[i].compile()
        weight_path = f"trained_models_for_test/policy_subgoal_{i}.h5"
        if path.exists(weight_path):
            print(f"Loading .h5 weights for subgoal {i} from {weight_path}")
            success = load_h5_to_hdqn(weight_path, agent_list[i].hdqn)
            if not success:
                print(f"Failed to load weights for subgoal {i}, training from scratch")
        else:
            print(f"No pre-trained weights found for subgoal {i}, training from scratch.")
        if i not in goal_to_train:
            agent_list[i].randomPlay = False
            agent_list[i].controllerEpsilon = 0.0

    option_learned = [i not in goal_to_train for i in range(Num_subgoal)]
    training_completed = False

    episodeCount = 0
    stepCount = 0

    option_t = [0] * Num_subgoal
    option_training_counter = [0] * Num_subgoal

    plantrace = []
    ro_table_lp = [(5, 2), (5, 0)]  # Example: from devilleft move(key) and move(lower_right_ladder)

    nS = 14
    nA = 6

    R_table = np.zeros((nS, nA))
    ro_table = np.zeros((nS, nA))
    explore = True
    converged = False

    generate_goal_file(0)
    cleanupconstraint()

    while episodeCount < EPISODE_LIMIT and stepCount < STEPS_LIMIT:
        print("\n\n### EPISODE", episodeCount, "###")
        env.restart()
        episodeSteps = 0
        replanned = False
        stateaction = []
        planquality = 0
        if len(ro_table_lp) > 0:
            generate_rovalue_from_table(env, ro_table_lp, ro_table)
        done = False
        allsubgoallearned = True

        # Plan generation and constraint management
        if explore:
            print("generate new plan...")
            oldplan = plantrace
            plantrace = generateplan()
            planabandoned = False
            if plantrace is None:
                print("No plan found at Episode", episodeCount)
                print("Continuing previous plan.")
                converged = True
                plantrace = oldplan

        goal_index = 0
        goal_not_found = False
        plan_failed = False
        # Only clear constraint at start of training, not per episode

        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode and goal_index < len(plantrace) - 1 and not goal_not_found and not planabandoned:

            goal = selectSubGoal(plantrace, goal_index)
            if not option_learned[goal]:
                allsubgoallearned = False

            state_ind, action_ind = obtainStateAction(plantrace, goal_index)

            if goal == -1:
                print("Subgoal not found for", plantrace[goal_index+1][2])
                # Update constraint when plan is abandoned due to bad subgoal
                updateconstraint(state_ind, action_ind)
                planabandoned = True
                goal_not_found = True
                explore = True
                break

            print('Current state and action:', plantrace[goal_index][2], state_ind, plantrace[goal_index][2], action_ind)
            print('Predicted subgoal:', plantrace[goal_index+1][2], 'Goal explanation:', goalExplain[goal])

            # Get initial state
            state = env.getStackedState()
            state_tensor = process_state_tensor(state)

            reachGoal = False
            die = False
            option_steps = 0
            option_reward = 0

            if option_learned[goal]:
                print("Option already learned, executing...")
            else:
                print("Option not learned yet, training...")

            while not reachGoal and not die and not env.isTerminal() and option_steps < 500:
                action = agent_list[goal].selectMove(state_tensor)
                reward = env.act(actionMap[action])
                next_state = env.getStackedState()
                next_state_tensor = process_state_tensor(next_state)

                reachGoal = env.goalReached(goal)
                # Debug print for distance-based reward and goal status
                distanceReward = env.distanceReward(-1, goal)
                print("→ Distance reward:", distanceReward, "Goal reached:", reachGoal)
                die = env.isTerminal()

                if not option_learned[goal]:
                    r = agent_list[goal].criticize(reachGoal, action, die, distanceReward, args.use_sparse_reward)
                    agent_list[goal].store(state_tensor.cpu().numpy(), action, r, next_state_tensor.cpu().numpy(), reachGoal or die)

                    if stepCount % agent_list[goal].trainFreq == 0:
                        # Fix: Pass stepCount to the update method
                        agent_list[goal].update(stepCount)

                state = next_state
                state_tensor = next_state_tensor
                option_steps += 1
                episodeSteps += 1
                stepCount += 1
                option_reward += reward

                if reachGoal:
                    print("Goal reached!")
                    subgoal_success_tracker[goal].append(1)
                    if len(subgoal_success_tracker[goal]) > 20:
                        subgoal_success_tracker[goal].pop(0)
                    break
                elif die:
                    print("Agent died!")
                    subgoal_success_tracker[goal].append(0)
                    if len(subgoal_success_tracker[goal]) > 20:
                        subgoal_success_tracker[goal].pop(0)
                    break

            if len(subgoal_success_tracker[goal]) >= 10:
                subgoal_trailing_performance[goal] = sum(subgoal_success_tracker[goal][-10:]) / 10.0
                if subgoal_trailing_performance[goal] >= 0.8 and not option_learned[goal]:
                    option_learned[goal] = True
                    print(f"Option {goal} learned with performance {subgoal_trailing_performance[goal]}")

            if reachGoal:
                goal_index += 1
                if goal_index == len(plantrace) - 1:
                    print("Plan completed successfully!")
                    break
            else:
                print("Failed to reach subgoal, abandoning plan")
                planabandoned = True
                plan_failed = True
                explore = True
                break

        episodeCount += 1
        print(f"Episode {episodeCount} completed in {episodeSteps} steps")
        print(f"Total steps: {stepCount}")
        print(f"Subgoal trailing performance: {subgoal_trailing_performance}")
        print(f"Option learned status: {option_learned}")

        if allsubgoallearned and not training_completed:
            print("All subgoals learned! Training completed.")
            training_completed = True

if __name__ == "__main__":
    main()
