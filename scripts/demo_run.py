#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from robot_env import RobotEnv
from stable_baselines3 import PPO

MODEL_PATH = os.path.expanduser(
    "~/projektarbeit/ros2_ws/src/autonomes_fahren_gazebo/training/models/ppo_turtlebot_850000_steps.zip")

env = RobotEnv()
model = PPO.load(MODEL_PATH, env=env)

# Fuer die Demo: volles Ziel (x=7), nicht die Curriculum-Zwischenstufe
env.set_goal(3.0)

NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    result = "ZIEL ERREICHT" if info.get("goal_reached", 0) == 1 else (
        "KOLLISION" if info.get("collided", 0) == 1 else "ABGEBROCHEN (Stuck/Zeit)")
    print(f"Episode {ep+1}: {result} | Schritte: {steps} | Reward: {total_reward:.1f} | max_x: {info.get('max_x', 'n/a')}")

env.close()
