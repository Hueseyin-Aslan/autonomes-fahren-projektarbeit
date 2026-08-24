#!/usr/bin/env python3
import sys
import os
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
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

env = RobotEnv()
env = Monitor(env, info_keywords=("max_x", "goal_reached", "collided", "stuck"))

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=MODEL_DIR,
    name_prefix="ppo_turtlebot"
)


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
    """Schaltet die Zielposition automatisch weiter, sobald die Erfolgsquote
    ueber ein Fenster von `window` Episoden den `success_threshold` erreicht."""

    def __init__(self, stages, success_threshold=0.7, window=50, verbose=1):
        super().__init__(verbose)
        self.stages = stages
        self.stage_idx = 0
        self.threshold = success_threshold
        self.window = window
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

        if len(self.success_buffer) >= self.window:
            rate = sum(self.success_buffer) / len(self.success_buffer)
            self.logger.record("curriculum/success_rate_window", rate)

            if rate >= self.threshold and self.stage_idx < len(self.stages) - 1:
                self.stage_idx += 1
                new_goal = self.stages[self.stage_idx]
                self.training_env.env_method("set_goal", new_goal)
                self.success_buffer.clear()
                if self.verbose:
                    print(f"\n=== CURRICULUM: Stufe erhoeht -> neues Ziel x={new_goal} "
                          f"(Erfolgsquote war {rate:.2f}) ===\n")
        return True


CURRICULUM_STAGES = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
)

print("Starte Training mit Curriculum...")
print(f"Stufen: {CURRICULUM_STAGES}")

model.learn(
    total_timesteps=250000,
    callback=CallbackList([
        checkpoint_callback,
        ProgressCallback(),
        CurriculumCallback(CURRICULUM_STAGES, success_threshold=0.7, window=50),
    ]),
    progress_bar=True,
)

final_path = os.path.join(MODEL_DIR, "ppo_turtlebot_final")
model.save(final_path)
print(f"Training abgeschlossen. Modell gespeichert unter: {final_path}")

env.close()
