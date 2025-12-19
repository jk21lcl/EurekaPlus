import logging
import os
import subprocess

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from eureka_plus import EurekaPlus
from utils.misc import block_until_training, set_freest_gpu
from utils.file_utils import load_tensorboard_logs

def prepare_cfg(cfg: DictConfig):
    # Prepare configuration
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Resolve root directories
    project_root = cfg.paths.project_root
    isaac_root = cfg.paths.isaac_root

    cfg.paths.train_script = f'{isaac_root}/train.py'
    cfg.paths.prompt_dir = f'{project_root}/utils/prompts'

    # Resolve file paths
    task = cfg.env.task
    suffix = cfg.suffix
    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{project_root}/envs/isaac') else 'dexterity'

    cfg.paths.base_task_file = f'{project_root}/envs/{env_parent}/{env_name}.py'
    cfg.paths.base_task_obs_file = f'{project_root}/envs/{env_parent}/{env_name}_obs.py'
    cfg.paths.generated_task_file = f"{isaac_root}/tasks/{env_name}_{suffix.lower()}.py"

    cfg.paths.base_task_config = f"{isaac_root}/cfg/task/{task}.yaml"
    cfg.paths.generated_task_config = f"{isaac_root}/cfg/task/{task}{suffix}.yaml"
    cfg.paths.base_train_config = f"{isaac_root}/cfg/train/{task}PPO.yaml"
    cfg.paths.generated_train_config = f"{isaac_root}/cfg/train/{task}{suffix}PPO.yaml"

    # Resolve output file paths
    cfg.paths.reward_iter_output = "env_iter{iter}_response{id}.txt"
    cfg.paths.eval_output = "reward_code_eval{i}.txt"

    cfg.paths.obs_file_copy = "env_init_obs.py"
    cfg.paths.generated_reward_copy = "env_iter{iter}_response{id}_rewardonly.py"
    cfg.paths.generated_task_file_copy = "env_iter{iter}_response{id}.py"
    
    cfg.paths.summary_figure = "summary.png"
    cfg.paths.summary_stats = "summary.npz"
    cfg.paths.eval_stats = "final_eval.npz"
    cfg.paths.message_log = "messages.json"

    return cfg

def evaluate_reward_code(cfg: DictConfig):
    # Evaluate the best reward code many times
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")

    eval_runs = []
    for i in range(cfg.num_eval):
        # set_freest_gpu()

        # Execute the python file with flags
        rl_filepath = cfg.paths.eval_output.format(i=i)
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(
                [
                    'python',
                    '-u',
                    cfg.paths.train_script,
                    'hydra/output=subprocess',
                    f'task={cfg.env.task}{cfg.suffix}',
                    f'wandb_activate={cfg.use_wandb}',
                    f'wandb_entity={cfg.wandb_username}',
                    f'wandb_project={cfg.wandb_project}',
                    f'headless={not cfg.capture_video}',
                    f'capture_video={cfg.capture_video}',
                    'force_render=False',
                    f'seed={i}',
                ],
                stdout=f,
                stderr=f,
            )
        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = cfg.paths.eval_output.format(i=i)
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "llm_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            llm_reward = np.array(tensorboard_logs["llm_reward"])
            reward_correlation = np.corrcoef(gt_reward, llm_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez(cfg.paths.eval_stats, reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    cfg = prepare_cfg(cfg)

    eureka_plus = EurekaPlus(cfg)
    eureka_plus.run()

    evaluate_reward_code(cfg)

if __name__ == "__main__":
    main()