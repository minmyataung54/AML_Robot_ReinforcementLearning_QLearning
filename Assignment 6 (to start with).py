#!/usr/bin/python3
"""Assignment 6 – Q-learning robot with simple, effective rewards."""

from pathlib import Path
import csv
import datetime as dt
import random

from pysimbotlib.core import PySimbotApp, Robot
from kivy.config import Config

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    print(f"matplotlib backend: {matplotlib.get_backend()}")
except Exception as e:
    print(f"matplotlib setup failed: {e}")
    plt = None

Config.set("kivy", "log_level", "info")

ALPHA = 0.5
GAMMA = 0.9
MAX_TICK = 100_000
FORWARD_STEP = 6
TURN_ANGLE = 20

STATE_SHAPE = (3, 2, 2, 2, 2, 2, 3)
Q = np.zeros(STATE_SHAPE, dtype=np.float32)

metrics = {
    "tick": [],
    "food_rate": [],
    "collision_rate": [],
}

LIVE_UPDATE_EVERY = 500
live_fig = None
live_axes = None
live_lines = None
live_enabled = False


def setup_live_plot() -> None:
    """Create live matplotlib figure if an interactive backend is available."""
    global live_fig, live_axes, live_lines, live_enabled

    if plt is None:
        print("matplotlib unavailable; live chart disabled.")
        live_enabled = False
        return

    backend = plt.get_backend()
    print(f"Attempting live plot with backend: {backend}")

    plt.ion()
    live_fig, live_axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    live_axes[0].set_title("Eat Rate")
    live_axes[0].set_ylabel("Eat Rate")
    live_axes[0].grid(True, alpha=0.3)

    live_axes[1].set_title("Collision Rate")
    live_axes[1].set_xlabel("Time (Steps)")
    live_axes[1].set_ylabel("Collision Rate")
    live_axes[1].grid(True, alpha=0.3)

    food_line = live_axes[0].plot([], [], color="green", linewidth=1.5)[0]
    collision_line = live_axes[1].plot([], [], color="red", linewidth=1.5)[0]
    live_lines = (food_line, collision_line)

    live_fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    live_enabled = True
    print("Live plot window opened.")


def update_live_plot() -> None:
    """Refresh live matplotlib lines with the latest metrics."""
    if not live_enabled or live_fig is None or live_lines is None:
        return
    if not metrics["tick"]:
        return

    ticks = metrics["tick"]
    food_rates = metrics["food_rate"]
    collision_rates = metrics["collision_rate"]

    live_lines[0].set_data(ticks, food_rates)
    live_lines[1].set_data(ticks, collision_rates)

    for ax in live_axes:
        ax.relim()
        ax.autoscale_view()

    live_fig.canvas.draw_idle()
    live_fig.canvas.flush_events()
    plt.pause(0.001)


def reset_learning() -> None:
    global Q
    Q = np.zeros(STATE_SHAPE, dtype=np.float32)
    for values in metrics.values():
        values.clear()
    if live_lines is not None:
        live_lines[0].set_data([], [])
        live_lines[1].set_data([], [])
        if live_axes is not None:
            for ax in live_axes:
                ax.relim()
                ax.autoscale_view()
            if live_fig is not None:
                live_fig.canvas.draw_idle()
                live_fig.canvas.flush_events()


def log_metrics(robot: Robot) -> None:
    tick = robot._sm.iteration
    if tick <= 0:
        return
    metrics["tick"].append(tick)
    metrics["food_rate"].append(robot.eat_count / tick)
    metrics["collision_rate"].append(robot.collision_count / tick)

    if tick % 1000 == 0:
        print(
            f"[Tick {tick:6d}] Eats: {robot.eat_count:4d} (rate {metrics['food_rate'][-1]:.4f}) | "
            f"Collisions: {robot.collision_count:4d} (rate {metrics['collision_rate'][-1]:.4f})"
        )

    if live_enabled and (len(metrics["tick"]) <= 50 or tick % LIVE_UPDATE_EVERY == 0):
        update_live_plot()


def export_results(_simbot) -> None:
    if not metrics["tick"]:
        print("No metrics captured; skipping export.")
        return

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    csv_path = out_dir / f"assignment6_metrics_{timestamp}.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tick", "food_rate", "collision_rate"])
        writer.writerows(zip(metrics["tick"], metrics["food_rate"], metrics["collision_rate"]))
    print(f"Metrics saved to {csv_path}")

    if plt is None:
        print("matplotlib unavailable; open the CSV for analysis.")
        return

    ticks = metrics["tick"]
    food_rates = metrics["food_rate"]
    collision_rates = metrics["collision_rate"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(ticks, food_rates, color="green", linewidth=1.5)
    axes[0].set_title("Eat Rate")
    axes[0].set_ylabel("Eat Rate")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ticks, collision_rates, color="red", linewidth=1.5)
    axes[1].set_title("Collision Rate")
    axes[1].set_xlabel("Time (Steps)")
    axes[1].set_ylabel("Collision Rate")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"assignment6_learning_curves_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Learning curves saved to {plot_path}")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show(block=True)
    else:
        print("Current matplotlib backend is non-interactive; open the saved PNG to view the charts.")


class RL_Robot(Robot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_state = None
        self.prev_action = None

    def update(self) -> None:
        # Get current sensor readings
        distances = self.distance()
        target = self.smell()
        
        # Adjust target angle if > 180
        if target > 180:
            target = target - 360
        
        # Discretize IR sensors
        IR = int(distances[0] < 40)
        LIR = int(distances[7] < 40)
        RIR = int(distances[1] < 40)
        LLIR = int(distances[6] < 40)
        RRIR = int(distances[2] < 40)
        
        # Discretize smell (3 buckets)
        if target > 30:
            SM = 2
        elif target < -30:
            SM = 0
        else:
            SM = 1
        
        # Q-learning update (if we have history)
        if self.prev_state is not None and self.prev_action is not None:
            s0, s1, s2, s3, s4, sm = self.prev_state
            at = self.prev_action
            
            # SIMPLE reward: only collision and eating
            if self.stuck:
                reward = -2.0
            elif self.just_eat:
                reward = 3.0  # Increased from 1.0
            else:
                reward = 0.0
            
            # Get Q-value for FORWARD action in current state
            q_forward = Q[0][IR][LIR][RIR][LLIR][RRIR][SM]
            
            # Q-learning update
            old_q = Q[at][s0][s1][s2][s3][s4][sm]
            Q[at][s0][s1][s2][s3][s4][sm] = old_q + ALPHA * (reward + GAMMA * q_forward - old_q)
        
        # Action selection (epsilon-greedy)
        if random.random() < 0.1:
            action = random.randint(0, 2)
        else:
            q0 = Q[0][IR][LIR][RIR][LLIR][RRIR][SM]
            q1 = Q[1][IR][LIR][RIR][LLIR][RRIR][SM]
            q2 = Q[2][IR][LIR][RIR][LLIR][RRIR][SM]
            
            # Break ties by preferring forward
            if q0 >= q1 and q0 >= q2:
                action = 0
            elif q1 >= q2:
                action = 1
            else:
                action = 2
        
        # Execute action
        if action == 0:
            self.move(FORWARD_STEP)
        elif action == 1:
            self.turn(-TURN_ANGLE)
        else:
            self.turn(TURN_ANGLE)
        
        # Store state and action for next update
        self.prev_state = (IR, LIR, RIR, LLIR, RRIR, SM)
        self.prev_action = action
        
        log_metrics(self)


def before_simulation(_simbot) -> None:
    reset_learning()
    setup_live_plot()
    print("Starting Q-learning with simple rewards...")
    print(f"Alpha: {ALPHA}, Gamma: {GAMMA}, Epsilon: 0.1")
    print("Rewards: Collision=-2.0, Eating=+3.0")


if __name__ == "__main__":
    app = PySimbotApp(
        robot_cls=RL_Robot,
        num_robots=1,
        max_tick=MAX_TICK,
        simulation_forever=False,
        customfn_before_simulation=before_simulation,
        customfn_after_simulation=export_results,
    )
    app.run()