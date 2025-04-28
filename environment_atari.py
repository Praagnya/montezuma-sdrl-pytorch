# Hoang M. Le
# California Institute of Technology
# Refactored for PyTorch Compatibility (no functional change)

from hyperparameters_new import *
import sys
import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ale_python_interface import ALEPyTorchInterface

logger = logging.getLogger(__name__)
np.random.seed(SEED)

class ALEEnvironment():

  def __init__(self, rom_file, args):
    self.ale = ALEPyTorchInterface()
    self.histLen = 4

    if args.display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.ale.setBool('sound', False)
      elif sys.platform.startswith('linux'):
        self.ale.setBool('sound', True)
      self.ale.setBool('display_screen', True)

    self.ale.setInt('frame_skip', args.frame_skip)
    self.ale.setFloat('repeat_action_probability', 0.0)
    self.ale.setBool('color_averaging', args.color_averaging)
    self.ale.setInt('random_seed', 0)
    self.ale.loadROM(rom_file)

    if args.minimal_action_set:
      self.actions = self.ale.getMinimalActionSet()
      logger.info("Using minimal action set with size %d" % len(self.actions))
    else:
      self.actions = self.ale.getLegalActionSet()
      logger.info("Using full action set with size %d" % len(self.actions))

    self.screen_width = args.screen_width
    self.screen_height = args.screen_height

    self.mode = "train"
    self.life_lost = False
    self.initSrcreen = self.getScreen()

    self.goalSet = []
    self.goalSet.append([[69, 68], [73, 71]])
    self.goalSet.append([[7, 41], [11, 45]])
    self.goalSet.append([[11, 68], [15, 71]])
    self.goalSet.append([[69, 68], [73, 71]])
    self.goalSet.append([[70, 20], [73, 35]])

    self.goalCenterLoc = []
    for goal in self.goalSet:
      goalCenter = [(goal[0][0] + goal[1][0]) / 2.0, (goal[0][1] + goal[1][1]) / 2.0]
      self.goalCenterLoc.append(goalCenter)

    self.agentOriginLoc = [42, 33]
    self.agentLastX = 42
    self.agentLastY = 33
    self.devilLastX = 0
    self.devilLastY = 0
    self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]
    self.histState = self.initializeHistState()

  def initializeHistState(self):
    s = self.getState()
    return np.concatenate([s, s, s, s], axis=0)

  def numActions(self):
    return len(self.actions)

  def resetGoalReach(self):
    self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]

  def restart(self):
    self.ale.reset_game()
    self.life_lost = False
    self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]
    for _ in range(19):
      self.act(0)
    self.histState = self.initializeHistState()
    self.agentLastX = self.agentOriginLoc[0]
    self.agentLastY = self.agentOriginLoc[1]

  def beginNextLife(self):
    self.life_lost = False
    self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]
    for _ in range(19):
      self.act(0)
    self.histState = self.initializeHistState()
    self.agentLastX = self.agentOriginLoc[0]
    self.agentLastY = self.agentOriginLoc[1]

  def act(self, action):
    lives = self.ale.lives()
    reward = self.ale.act(self.actions[action])
    self.life_lost = (lives != self.ale.lives())
    currState = self.getState()
    self.histState = np.concatenate((self.histState[1:], currState), axis=0)
    return reward

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()

    # Convert tensor -> numpy if needed
    if isinstance(screen, torch.Tensor):
        screen = screen.cpu().numpy()

    if screen is None or not isinstance(screen, np.ndarray) or screen.size == 0:
        print(f"⚠️ Invalid screen, type: {type(screen)}")
        screen = np.zeros((210, 160), dtype=np.uint8)

    resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    return resized

  def getScreenRGB(self):
    screen = self.ale.getScreenRGB()
    if isinstance(screen, torch.Tensor):
        screen = screen.cpu().numpy()
    resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    return resized  

  def getAgentLoc(self, img):
    man = [200, 72, 72]
    mask = np.zeros_like(img)
    mask[:, :, 0] = man[0]
    mask[:, :, 1] = man[1]
    mask[:, :, 2] = man[2]
    diff = img - mask
    indxs = np.where(diff == 0)
    if len(indxs[0]) == 0:
      mean_x = self.agentLastX
      mean_y = self.agentLastY
    else:
      mean_y = np.sum(indxs[0]) / len(indxs[0])
      mean_x = np.sum(indxs[1]) / len(indxs[1])
    self.agentLastX = mean_x
    self.agentLastY = mean_y
    return mean_x, mean_y

  def getDevilLoc(self, img):
    devilColor = [236, 236, 236]
    mask = np.zeros_like(img)
    mask[:, :, 0] = devilColor[0]
    mask[:, :, 1] = devilColor[1]
    mask[:, :, 2] = devilColor[2]
    diff = img - mask
    indxs = np.where(diff == 0)
    if len(indxs[0]) == 0:
      mean_x = self.devilLastX
      mean_y = self.devilLastY
    else:
      mean_y = np.sum(indxs[0]) / len(indxs[0])
      mean_x = np.sum(indxs[1]) / len(indxs[1])
    self.devilLastX = mean_x
    self.devilLastY = mean_y
    return mean_x, mean_y

  def distanceReward(self, lastGoal, goal):
    img = self.getScreenRGB()
    if lastGoal == -1:
      lastGoalCenter = self.agentOriginLoc
    else:
      lastGoalCenter = self.goalCenterLoc[lastGoal]
    goalCenter = self.goalCenterLoc[goal]
    agentX, agentY = self.getAgentLoc(img)
    dis = np.linalg.norm(np.array(goalCenter) - np.array([agentX, agentY]))
    disLast = np.linalg.norm(np.array(lastGoalCenter) - np.array([agentX, agentY]))
    disGoals = np.linalg.norm(np.array(goalCenter) - np.array(lastGoalCenter))
    return (disLast - dis) / disGoals

  def getState(self):
    screen = self.ale.getScreenGrayscale()

    if isinstance(screen, torch.Tensor):
        screen = screen.cpu().numpy()

    if screen is None or not isinstance(screen, np.ndarray) or screen.size == 0:
        screen = np.zeros((210, 160), dtype=np.uint8)

    resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    return np.expand_dims(resized, axis=0).astype(np.float32)

  def getStackedState(self):
    return self.histState.astype(np.float32)

  def isTerminal(self):
    if self.mode == 'train':
      return self.ale.game_over() or self.life_lost
    return self.ale.game_over()

  def isGameOver(self):
    return self.ale.game_over()

  def isLifeLost(self):
    return self.life_lost

  def reset(self):
    self.ale.reset_game()
    self.life_lost = False

  def goalReached(self, goal):
    subset = [0, 2, 3, 4, 6]
    if goal in subset:
      goal_index = subset.index(goal)
      goalPosition = self.goalSet[goal_index]
      goalScreen = self.initSrcreen
      stateScreen = self.getScreen()
      count = 0
      for y in range(goalPosition[0][0], goalPosition[1][0]):
        for x in range(goalPosition[0][1], goalPosition[1][1]):
          if goalScreen[x][y] != stateScreen[x][y]:
            count += 1
      if float(count) / 30 > 0.3:
        self.reachedGoal[goal] = 1
        return True
    if goal == 1:
      return self.detect_left_ladder()
    if goal == 5:
      return self.original_location_reached()
    return False

  def detect_right_ladder(self):
    return self.detect_ladder(0)

  def detect_left_ladder(self):
    return self.detect_ladder(2)

  def detect_ladder(self, index):
    goalPosition = self.goalSet[index]
    goalScreen = self.initSrcreen
    stateScreen = self.getScreen()
    count = 0
    for y in range(goalPosition[0][0], goalPosition[1][0]):
      for x in range(goalPosition[0][1], goalPosition[1][1]):
        if goalScreen[x][y] != stateScreen[x][y]:
          count += 1
    return float(count) / 30 > 0.3

  def original_location_reached(self):
    img = self.getScreenRGB()
    x, y = self.getAgentLoc(img)
    return abs(x-42) <= 2 and abs(y-33) <= 2

  def goalNotReachedBefore(self, goal):
    return self.reachedGoal[goal] == 0