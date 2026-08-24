#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from robot_env import RobotEnv

env = RobotEnv()

for episode in range(2):
    obs, info = env.reset()
    print(f"\n=== Episode {episode + 1} gestartet ===")
    print(f"Start-Observation (LiDAR min/max): {obs[:24].min():.2f} / {obs[:24].max():.2f}")
    print(f"Start-Position: x={obs[24]:.2f}, y={obs[25]:.2f}")

    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(f"  Step {step}: action={action}, reward={reward:.2f}, "
                  f"x={obs[24]:.2f}, terminated={terminated}")

        if terminated or truncated:
            print(f"  Episode beendet nach {step + 1} Schritten "
                  f"(terminated={terminated}, truncated={truncated})")
            break

    print(f"Gesamt-Reward: {total_reward:.2f}")

env.close()
print("\nTest abgeschlossen.")
