import gymnasium as gym
from environments.curriculum_env import S0100Env
from environments.curriculum_env import FrameStack
from Agents.SAC_curriculum import SACAgent
import os
import re
import argparse
import torch

def make_env(max_episode_steps=400, success_hold_steps=3, task_stage=1):
    def _init():
        env = S0100Env(
            render_mode=None,
            max_episode_steps=max_episode_steps,
            success_hold_steps=success_hold_steps,
            task_stage=task_stage,
        )
        # env = FrameStack(env, num_stack=3)
        return env
    return _init

def _latest_actor_step(models_dir):
    actor_dir = os.path.join(models_dir, "Actor")
    if not os.path.isdir(actor_dir):
        return None

    latest = None
    for name in os.listdir(actor_dir):
        match = re.match(r"^(\d+)", name)
        if match is None:
            continue
        step = int(match.group(1))
        latest = step if latest is None else max(latest, step)
    return latest


def run_stage(args, device, models_dir, task_stage, resume_step):
    num_envs = args.num_envs
    env = gym.vector.AsyncVectorEnv(
        [make_env(args.episode_steps, args.success_hold_steps, task_stage) for _ in range(num_envs)]
    )

    model = SACAgent(
        env,
        device=device,
        timesteps=args.timesteps,
        use_images=False,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
    )

    start_timestep = 0
    if resume_step is not None:
        model.load_checkpoint(models_dir, resume_step, load_critic=args.resume_load_critic)
        start_timestep = resume_step

    try:
        model.train(models_dir, save_timesteps=args.save_timesteps, start_timestep=start_timestep)
    finally:
        env.close()

    return _latest_actor_step(models_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-step", type=int, default=None, help="Load actor/critic checkpoint from this timestep")
    parser.add_argument("--timesteps", type=int, default=30_000_000, help="Number of additional training steps to run")
    parser.add_argument("--save-timesteps", type=int, default=12_500, help="Checkpoint save interval")
    parser.add_argument("--episode-steps", type=int, default=400, help="Episode horizon handled by S0100Env")
    parser.add_argument("--success-hold-steps", type=int, default=1, help="Consecutive success steps before terminate")
    parser.add_argument("--task-stage", type=int, choices=[1, 2, 3, 4], default=1, help="Curriculum stage: 1=reach, 2=slide, 3=pick-place, 4=stack")
    parser.add_argument("--num-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--replay-size", type=int, default=1_000_000, help="Replay capacity in transitions")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--updates-per-step", type=int, default=2, help="Gradient updates per env step")
    parser.add_argument("--curriculum-stages", type=int, nargs="+", choices=[1, 2, 3, 4], default=None, help="Optional stage sequence, e.g. --curriculum-stages 1 2 3 4")
    parser.add_argument("--resume-load-critic", action="store_true", help="When resuming, also load critic. Default is actor-only transfer.")
    args = parser.parse_args()

    models_dir = "models/SAC_stacking"
    log_dir = "SAC_logs_with_rand_sphere"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.curriculum_stages:
        resume_step = args.resume_step
        for stage in args.curriculum_stages:
            print(f"\n=== Curriculum stage {stage} ===")
            resume_step = run_stage(
                args=args,
                device=device,
                models_dir=models_dir,
                task_stage=stage,
                resume_step=resume_step,
            )
            if resume_step is None:
                raise RuntimeError("No actor checkpoint found after stage run; cannot continue curriculum.")
    else:
        _ = run_stage(
            args=args,
            device=device,
            models_dir=models_dir,
            task_stage=args.task_stage,
            resume_step=args.resume_step,
        )

if __name__ == '__main__':
    main()