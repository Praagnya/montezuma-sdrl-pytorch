#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PyTorch implementation of subgoal testing script
# Converted from TensorFlow implementation by Manus
# Original credits:
# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================

import argparse
import sys
import time
import numpy as np
import torch
from collections import namedtuple, deque
from environment_atari import ALEEnvironment
from hybrid_rl_il_agent_atari import Agent
from simple_net import Net
from PIL import Image
from tensorboard import TensorboardVisualizer
import os
import time
from subgoal_check import isSubgoalReached

nb_Action = 8
maxStepsPerEpisode = 10000
np.random.seed(0)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']

    ############## -- DML modified -- ###########
    goalExplain = ['lower right ladder', 'jump to the left of devil', 'key', 'lower left ladder',
                   'lower right ladder', 'central high platform', 'right door']  # 7
    ################# -- end -- ###########

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=True)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--color_averaging", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--minimal_action_set", type=str2bool, default=False)
    parser.add_argument("--screen_width", type=int, default=84)
    parser.add_argument("--screen_height", type=int, default=84)
    parser.add_argument("--load_weight", type=str2bool, default=False)
    parser.add_argument("--use_sparse_reward", type=str2bool, default=True)
    parser.add_argument("--max_episodes", type=int, default=80000)
    args = parser.parse_args()
    env = ALEEnvironment(args.game, args)
    
    # Initialize TensorBoard visualizer
    visualizer = TensorboardVisualizer()
    visualizer.initialize(logdir="./tensorboard_logs")
    
    # Initialize network and agent
    episodeCount = 0

    # Create PyTorch networks for each subgoal
    firstNet = Net()
    secondNet = Net()
    thirdNet = Net()
    fourthNet = Net()
    fifthNet = Net()
    sixthNet = Net()
    seventhNet = Net()
    
    # Load weights for each network
    firstNet.loadWeight(0)
    secondNet.loadWeight(1)
    thirdNet.loadWeight(2)
    fourthNet.loadWeight(3)
    fifthNet.loadWeight(4)
    sixthNet.loadWeight(5)
    seventhNet.loadWeight(6)

    # Main loop
    while episodeCount < args.max_episodes:
        print("\n\n### EPISODE " + str(episodeCount) + "###")
        # Restart the game
        env.restart()
        episodeSteps = 0

        # Log episode start
        visualizer.add_entry(episodeCount, "episode/start", episodeCount)

        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            stateLastGoal = env.getStackedState()

            # First goal
            goal = 0
            print('predicted subgoal is: ' + goalExplain[goal])
            
            # Log current goal
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = firstNet.selectMove(state, goal)

                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                
                # Log action
                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                
                # Log rewards
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)

                episodeSteps += 1
                nextState = env.getStackedState()
                
                # Optionally log state as image (commented out to avoid overhead)
                # if episodeSteps % 10 == 0:
                #     # Convert state to image format for visualization
                #     state_img = np.mean(nextState, axis=2) * 255  # Convert to grayscale
                #     state_img = state_img.astype(np.uint8)
                #     visualizer.add_entry(episodeSteps, "state/current", state_img, image=True)
                                                            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
            else:
                break
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()

            # Second goal
            goal = 1
            print('predicted subgoal is: ' + goalExplain[goal])
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = secondNet.selectMove(state, goal)
                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                print()
                
                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)
                
                episodeSteps += 1                
                nextState = env.getStackedState()

            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
            else:
                break

            # Third goal
            goal = 2
            print('predicted subgoal is: ' + goalExplain[goal])
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = thirdNet.selectMove(state, goal)
                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                print()

                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)
                
                episodeSteps += 1
                nextState = env.getStackedState()
                                                
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
            else:
                break

            # Fourth goal
            goal = 3
            print('predicted subgoal is: ' + goalExplain[goal])
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = fourthNet.selectMove(state, goal)
                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                print()

                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)
                
                episodeSteps += 1
                nextState = env.getStackedState()

            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
            else:
                break

            # Fifth goal
            goal = 4
            print('predicted subgoal is: ' + goalExplain[goal])
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = fifthNet.selectMove(state, goal)
                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                print()

                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)
                
                episodeSteps += 1
                nextState = env.getStackedState()

            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
            else:
                break

            # Sixth goal
            goal = 5
            print('predicted subgoal is: ' + goalExplain[goal])
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = sixthNet.selectMove(state, goal)
                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                print()

                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)
                
                episodeSteps += 1
                nextState = env.getStackedState()

            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
            else:
                break

            # Seventh goal
            goal = 6
            print('predicted subgoal is: ' + goalExplain[goal])
            visualizer.add_entry(episodeSteps, "goal/current", goal)
            
            while not env.isTerminal() and not isSubgoalReached(env, goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = seventhNet.selectMove(state, goal)
                print("episode step: ", episodeSteps)
                print("action chosen:", actionExplain[action])
                print()

                visualizer.add_entry(episodeSteps, "action/selected", action)
                
                externalRewards = env.act(actionMap[action])
                visualizer.add_entry(episodeSteps, "reward/external", externalRewards)
                
                episodeSteps += 1
                nextState = env.getStackedState()

            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                visualizer.add_entry(episodeSteps, "goal/reached", goal)
                
                # Final actions after reaching the last goal
                if goal == 6:
                    for i in range(15):
                        env.act(3)  # right
                    for i in range(15):
                        env.act(0)  # no action
                    break
            else:
                break
        
        # Log episode end
        visualizer.add_entry(episodeCount, "episode/steps", episodeSteps)
        episodeCount += 1
        
        # For testing purposes, break after a few episodes
        if episodeCount >= 10000:  # Run longer for testing
            break
    
    # Close visualizer
    visualizer.close()
    print("Testing completed successfully!")

if __name__ == "__main__":
    main()
