#!/usr/bin/env python3
import re
import sys
import os
import glob
sys.path.append(os.path.dirname(__file__))

from robot_env import RobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from collections import deque
from stable_baselines3.common.monitor import Monitor

MODEL_DIR = os.path.expanduser(
    "~/projektarbeit/ros2_ws/src/autonomes_fahren_gazebo/training/models")
LOG_DIR = os.path.expanduser(
    "~/projektarbeit/ros2_ws/src/autonomes_fahren_gazebo/training/logs")

# Fester Checkpoint: bewusst gewaehlter Stufe-2-Stand VOR dem Kollaps auf Stufe 3
latest_checkpoint = os.path.join(MODEL_DIR, "ppo_turtlebot_850000_steps.zip")
print(f"Setze fort mit: {latest_checkpoint}")

env = RobotEnv()
env = Monitor(env, info_keywords=("max_x", "goal_reached", "collided", "stuck"))


class ProgressCallback(BaseCallback):
    def __init__(self, window=50, verbose=0):
        super().__init__(verbose)
        self.max_x_buffer = deque(maxlen=window)
        self.goal_buffer = deque(maxlen=window)
        self.collided_buffer = deque(maxlen=window)
        self.stuck_buffer = deque(maxlen=window)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                if "max_x" in ep:
                    self.max_x_buffer.append(ep["max_x"])
                if "goal_reached" in ep:
                    self.goal_buffer.append(ep["goal_reached"])
                if "collided" in ep:
                    self.collided_buffer.append(ep["collided"])
                if "stuck" in ep:
                    self.stuck_buffer.append(ep["stuck"])
        if self.max_x_buffer:
            self.logger.record("rollout/max_x_mean", sum(self.max_x_buffer) / len(self.max_x_buffer))
        if self.goal_buffer:
            self.logger.record("rollout/success_rate", sum(self.goal_buffer) / len(self.goal_buffer))
        if self.collided_buffer:
            self.logger.record("rollout/collision_rate", sum(self.collided_buffer) / len(self.collided_buffer))
        if self.stuck_buffer:
            self.logger.record("rollout/stuck_rate", sum(self.stuck_buffer) / len(self.stuck_buffer))
        return True


class CurriculumCallback(BaseCallback):
    """Feinere Stufen, strengere Schwelle (0.80 ueber 100 Episoden),
    und erst nach 3 aufeinanderfolgenden Bestaetigungen wechseln."""

    def __init__(self, stages, start_stage_idx=0, success_threshold=0.80,
                 window=100, required_consecutive=3, verbose=1):
        super().__init__(verbose)
        self.stages = stages
        self.stage_idx = start_stage_idx
        self.threshold = success_threshold
        self.window = window
        self.required_consecutive = required_consecutive
        self.consecutive_passes = 0
        self.success_buffer = deque(maxlen=window)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                if "goal_reached" in ep:
                    self.success_buffer.append(ep["goal_reached"])

        current_goal = self.stages[self.stage_idx]
        self.logger.record("curriculum/goal_x", current_goal)
        self.logger.record("curriculum/stage_idx", self.stage_idx)
        self.logger.record("curriculum/window_episodes", len(self.success_buffer))

        if len(self.success_buffer) >= self.window:
            rate = sum(self.success_buffer) / len(self.success_buffer)
            self.logger.record("curriculum/success_rate_window", rate)

            if rate >= self.threshold:
                self.consecutive_passes += 1
            else:
                self.consecutive_passes = 0
            self.logger.record("curriculum/consecutive_passes", self.consecutive_passes)

            if (self.consecutive_passes >= self.required_consecutive
                    and self.stage_idx < len(self.stages) - 1):
                self.stage_idx += 1
                new_goal = self.stages[self.stage_idx]
                self.training_env.envs[0].unwrapped.set_goal(new_goal)
                self.success_buffer.clear()
                self.consecutive_passes = 0
                if self.verbose:
                    print(f"\n=== CURRICULUM: Stufe erhoeht -> neues Ziel x={new_goal} "
                          f"(Erfolgsquote war {rate:.2f}, bestaetigt) ===\n")
        return True


CURRICULUM_STAGES = [-3.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
START_STAGE_IDX = 4  # entspricht x=1.0, unser verifizierter guter Stand

model = PPO.load(latest_checkpoint, env=env)
model.learning_rate = 1e-4
model.lr_schedule = lambda _: 1e-4

env.unwrapped.set_goal(CURRICULUM_STAGES[START_STAGE_IDX])
print(f"Ziel gesetzt auf Stufe {START_STAGE_IDX}: x={CURRICULUM_STAGES[START_STAGE_IDX]}")
print(f"Learning Rate reduziert auf: 1e-4")

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=MODEL_DIR,
    name_prefix="ppo_turtlebot"
)

ADDITIONAL_TIMESTEPS = 200000
print(f"Trainiere {ADDITIONAL_TIMESTEPS} weitere Schritte...")
model.learn(
    total_timesteps=ADDITIONAL_TIMESTEPS,
    callback=CallbackList([
        checkpoint_callback,
        ProgressCallback(),
        CurriculumCallback(CURRICULUM_STAGES, start_stage_idx=START_STAGE_IDX,
                            success_threshold=0.80, window=100, required_consecutive=3),
    ]),
    progress_bar=True,
    reset_num_timesteps=False,
)

final_path = os.path.join(MODEL_DIR, "ppo_turtlebot_final")
model.save(final_path)
print(f"Training abgeschlossen. Modell gespeichert unter: {final_path}")

env.close()
