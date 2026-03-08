#!/usr/bin/env python3
import argparse
import sys
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", default="cuda:0")
    args, _ = parser.parse_known_args()

    # Start Kit FIRST
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, device=args.device)
    simulation_app = app_launcher.app

    try:
        try:
            import gymnasium as gym
            import torch
            import project_spiderbot  # noqa: F401 (registers task ids)
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

            env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
            env = gym.make(args.task, cfg=env_cfg)

            env.reset()
            print(f"[smoke] action_space={env.action_space}", flush=True)

            for i in range(args.num_steps):
                a_np = env.action_space.sample()
                a = torch.tensor(a_np, device=args.device, dtype=torch.float32)
                if a.ndim == 1:
                    a = a.unsqueeze(0)
                env.step(a)
                if (i + 1) % 10 == 0:
                    print(f"[smoke] step {i+1}/{args.num_steps}", flush=True)

            env.close()
            print("[smoke] DONE", flush=True)

        except Exception as e:
            print("\n[smoke] EXCEPTION:", repr(e), flush=True)
            traceback.print_exc()
            sys.exit(2)

    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
