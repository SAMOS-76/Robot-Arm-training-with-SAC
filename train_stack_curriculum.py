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


def _should_transfer_critic(prev_stage, next_stage):
    return prev_stage == 2 and next_stage == 3


def _build_stage_runner(args, device, task_stage):
    stage_hold_steps = args.stack_success_hold_steps if task_stage == 4 else args.success_hold_steps
    stage_success_threshold = args.stack_success_threshold if task_stage == 4 else args.base_success_threshold
    num_envs = args.num_envs
    env = gym.vector.AsyncVectorEnv(
        [make_env(args.episode_steps, stage_hold_steps, task_stage) for _ in range(num_envs)]
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

    # Need to vary warmup and learning steps depending on stage cause later stages need less warmup 
    if task_stage == 1:
        warmup_scale = 1.0
    elif task_stage in (2, 3):
        warmup_scale = args.stage23_warmup_scale
    else:
        warmup_scale = args.stage4_warmup_scale

    model.random_explore_steps = max(args.batch_size, int(model.random_explore_steps * warmup_scale))
    model.learning_steps = max(args.batch_size, int(model.learning_steps * warmup_scale))

    print(
        f"Stage {task_stage} schedule | "
        f"random_explore_steps={model.random_explore_steps} | "
        f"learning_steps={model.learning_steps}"
    )

    return env, model, stage_success_threshold


def run_stage_attempt(args, model, models_dir, start_timestep, stage_success_threshold):
    stage_solved = model.train(
        models_dir,
        save_timesteps=args.save_timesteps,
        start_timestep=start_timestep,
        target_success_rate=stage_success_threshold,
    )

    latest_step = _latest_actor_step(models_dir)
    if latest_step is None:
        raise RuntimeError("No actor checkpoint found after stage run; cannot continue curriculum.")

    return latest_step, stage_solved

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
    parser.add_argument("--resume-load-critic", action="store_true", help="When resuming first stage directly, also load critic. Default is actor-only transfer.")
    parser.add_argument("--base-success-threshold", type=float, default=0.90, help="Solved threshold for stages 1-3")
    parser.add_argument("--stack-success-threshold", type=float, default=0.80, help="Solved threshold for stage 4")
    parser.add_argument("--stack-success-hold-steps", type=int, default=5, help="Consecutive success steps required in stage 4")
    parser.add_argument("--stage23-warmup-scale", type=float, default=0.50, help="Scale factor for random_explore_steps and learning_steps in stages 2-3")
    parser.add_argument("--stage4-warmup-scale", type=float, default=0.35, help="Scale factor for random_explore_steps and learning_steps in stage 4")
    parser.add_argument("--max-stage-attempts", type=int, default=0, help="0 means unlimited attempts per stage")
    args = parser.parse_args()

    models_dir = "models/SAC_stacking"
    log_dir = "SAC_logs_with_rand_sphere"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.curriculum_stages:
        resume_step = args.resume_step
        prev_stage = None

        for stage in args.curriculum_stages:
            env, model, stage_success_threshold = _build_stage_runner(args, device, stage)

            try:
                attempt = 0

                if resume_step is not None:
                    if prev_stage is None:
                        load_critic_on_resume = args.resume_load_critic
                    else:
                        load_critic_on_resume = _should_transfer_critic(prev_stage, stage)

                    model.load_checkpoint(models_dir, resume_step, load_critic=load_critic_on_resume)

                start_timestep = resume_step if resume_step is not None else 0

                while True:
                    print(f"\n=== Curriculum stage {stage} | attempt {attempt + 1} ===")
                    resume_step, stage_solved = run_stage_attempt(
                        args=args,
                        model=model,
                        models_dir=models_dir,
                        start_timestep=start_timestep,
                        stage_success_threshold=stage_success_threshold,
                    )
                    if stage_solved:
                        break

                    attempt += 1
                    if args.max_stage_attempts > 0 and attempt >= args.max_stage_attempts:
                        print(f"Stage {stage} unsolved after {attempt} attempt(s). Stopping.")
                        return

                    start_timestep = resume_step
                    print(f"Stage {stage} not solved yet. Continuing same stage from step {resume_step}...")
            finally:
                env.close()

            prev_stage = stage
    else:
        env, model, stage_success_threshold = _build_stage_runner(args, device, args.task_stage)

        try:
            start_timestep = 0
            if args.resume_step is not None:
                model.load_checkpoint(models_dir, args.resume_step, load_critic=args.resume_load_critic)
                start_timestep = args.resume_step

            _ = run_stage_attempt(
                args=args,
                model=model,
                models_dir=models_dir,
                start_timestep=start_timestep,
                stage_success_threshold=stage_success_threshold,
            )
        finally:
            env.close()

if __name__ == '__main__':
    main()
