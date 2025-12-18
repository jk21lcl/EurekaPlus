import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml
from omegaconf import DictConfig
from openai import OpenAI
from openai.types.chat.chat_completion import Choice

from stats_manager import IterationStats, RunStats, StatsManager
from utils.misc import block_until_training, filter_traceback, set_freest_gpu
from utils.file_utils import load_tensorboard_logs
from utils.extract_task_code import file_to_string, get_function_signature

class Prompts:
    def __init__(self, prompt_dir: str):
        for filename in os.listdir(prompt_dir):
            if filename.endswith('.txt'):
                var_name = filename[:-4]
                file_path = os.path.join(prompt_dir, filename)
                setattr(self, var_name, file_to_string(file_path))

class EurekaPlus:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.task = cfg.env.task
        self.task_description = cfg.env.description
        self.env_name = cfg.env.env_name.lower()
        self.suffix = cfg.suffix
        self.model = cfg.model
        self.create_config_file()

        base_task_code_string = file_to_string(cfg.paths.base_task_file)
        self.task_code_string = base_task_code_string.replace(self.task, self.task+self.suffix)
        self.task_obs_code_string = file_to_string(cfg.paths.base_task_obs_file)
        shutil.copy(cfg.paths.base_task_obs_file, cfg.paths.obs_file_copy)

        self.prompts = Prompts(cfg.paths.prompt_dir)
        initial_system = self.prompts.initial_system.format(task_reward_signature_string=self.prompts.reward_signature) + self.prompts.code_output_tip
        initial_user = self.prompts.initial_user.format(task_obs_code_string=self.task_obs_code_string, task_description=self.task_description)
        self.messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
        self.stats_manager = StatsManager()

        logging.info(f"Workspace: {Path.cwd()}")
        logging.info(f"Project Root: {self.cfg.paths.project_root}")
        logging.info(f"Using LLM: {self.model}")
        logging.info("Task: " + self.task)
        logging.info("Task description: " + self.task_description)

    def create_config_file(self):
        """ Create generated task and training YAML configuration files """
        # Create task YAML file 
        with open(self.cfg.paths.base_task_config, 'r') as yamlfile:
            data = yaml.safe_load(yamlfile)

        # Modify the "name" field
        data['name'] = f'{self.task}{self.suffix}'
        data['env']['env_name'] = f'{self.env_name}_{self.suffix.lower()}'
        
        # Write the new YAML file
        with open(self.cfg.paths.generated_task_config, 'w') as new_yamlfile:
            yaml.safe_dump(data, new_yamlfile)

        # Create training YAML file
        with open(self.cfg.paths.base_train_config, 'r') as yamlfile:
            data = yaml.safe_load(yamlfile)

        # Modify the "name" field
        data['params']['config']['name'] = data['params']['config']['name'].replace(self.task, f'{self.task}{self.suffix}')

        # Write the new YAML file
        with open(self.cfg.paths.generated_train_config, 'w') as new_yamlfile:
            yaml.safe_dump(data, new_yamlfile)

    def generate_candidates(self, iter: int) -> List[Choice]:
        """ Generate multiple reward code candidates from LLM """
        responses: List[Choice] = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = self.cfg.sample if "gpt-3.5" in self.model else 4

        logging.info(f"Iteration {iter}: Generating {self.cfg.sample} samples with {self.cfg.model}")

        while True:
            if total_samples >= self.cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=self.cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        for id in range(self.cfg.sample):
            logging.info(f"Iteration {iter}: Generated Sample {id}:\n " + responses[id].message.content + "\n")

        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        return responses

    def post_process(self, iter: int, response_id: int, response_cur: str) -> str:
        """
        Post-process the generated code string, returns the modified code string.
        """
        logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

        # Regex patterns to extract python code enclosed in LLM response
        patterns = [
            r'```python(.*?)```',
            r'python(.*?)```',
            r'```py(.*?)```',
            r'```(.*?)```',
        ]
        for pattern in patterns:
            code_string = re.search(pattern, response_cur, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
                break
        code_string = response_cur if not code_string else code_string

        # Remove unnecessary imports
        lines = code_string.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                code_string = "\n".join(lines[i:])
                break
        
        # Add @torch.jit.script decorator above each function definition if not present
        lines = code_string.split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if i == 0 or not lines[i-1].strip().startswith("@torch.jit.script"):
                    new_lines.append("@torch.jit.script")
            new_lines.append(line)
        code_string = "\n".join(new_lines)
        
        with open(self.cfg.paths.generated_reward_copy.format(iter=iter, id=response_id), 'w') as file:
            file.writelines(code_string + '\n')
        
        return code_string
    
    def integrate_code_into_env(self, iter: int, response_id: int, code_string: str) -> bool:
        """
        Integrate the generated reward code into the environment code and save to file.
        Returns the success status.
        """
        # Add the Eureka Reward Signature to the environment code
        try:
            llm_reward_signature, input_lst = get_function_signature(code_string)
        except Exception as e:
            logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature! Error: {e}")
            return False
        
        reward_signature = [
            f"self.rew_buf[:], self.rew_dict = {llm_reward_signature}",
            f"self.extras['llm_reward'] = self.rew_buf.mean()",
            f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
        ]
        indent = " " * 8
        reward_signature = "\n".join([indent + line for line in reward_signature])
        if "def compute_reward(self)" in self.task_code_string:
            task_code_string_iter = self.task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
        elif "def compute_reward(self, actions)" in self.task_code_string:
            task_code_string_iter = self.task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
        else:
            raise NotImplementedError

        # Save the new environment code when the output contains valid code string!
        with open(self.cfg.paths.generated_task_file, 'w') as file:
            file.writelines(task_code_string_iter + '\n')
            file.writelines("from typing import Tuple, Dict" + '\n')
            file.writelines("import math" + '\n')
            file.writelines("import torch" + '\n')
            file.writelines("from torch import Tensor" + '\n')
            file.writelines(code_string + '\n')

        # Copy the generated environment code to hydra output directory for bookkeeping
        shutil.copy(self.cfg.paths.generated_task_file, self.cfg.paths.generated_task_file_copy.format(iter=iter, id=response_id))
        
        return True

    def launch_rl_training(self, iter: int, response_id: int) -> subprocess.Popen:
        """ Launch RL training subprocess with the generated reward code """
        # Find the freest GPU to run GPU-accelerated RL
        # set_freest_gpu()

        # Execute the python file with flags
        rl_filepath = self.cfg.paths.reward_iter_output.format(iter=iter, id=response_id)
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(
                [
                    'python',
                    '-u',
                    self.cfg.paths.train_script,
                    'hydra/output=subprocess',
                    f'task={self.task}{self.suffix}',
                    f'wandb_activate={self.cfg.use_wandb}',
                    f'wandb_entity={self.cfg.wandb_username}',
                    f'wandb_project={self.cfg.wandb_project}',
                    f'headless={not self.cfg.capture_video}',
                    f'capture_video={self.cfg.capture_video}',
                    'force_render=False',
                    f'max_iterations={self.cfg.max_iterations}',
                ],
                stdout=f,
                stderr=f,
            )
        block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)

        return process

    def gather_result(self, iter: int, response_id: int, rl_run: Optional[subprocess.Popen]) -> bool:
        """
        Gather RL training results and provide feedback.
        Returns success status.
        """
        success = False
        
        # initialize RunStats for this run
        code_path = self.cfg.paths.generated_task_file_copy.format(iter=iter, id=response_id)
        run_stats = RunStats(iteration=iter, response_id=response_id, code_path=code_path)
        feedback = ""

        rl_filepath = self.cfg.paths.reward_iter_output.format(iter=iter, id=response_id)
        
        if rl_run is None or not os.path.exists(rl_filepath):
            feedback += self.prompts.execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
            feedback += self.prompts.code_output_tip
            run_stats.feedback = feedback
            self.stats_manager.add_run(run_stats)
            return False

        rl_run.communicate()
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        
        traceback_msg = filter_traceback(stdout_str)
        if traceback_msg == '':
            # If RL execution has no error, provide policy statistics feedback
            success = True
            lines = stdout_str.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Tensorboard Directory:'):
                    break 
            tensorboard_logdir = line.split(':')[-1].strip() 
            tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
            max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
            epoch_freq = max(int(max_iterations // 10), 1)
            feedback += self.prompts.policy_feedback.format(epoch_freq=epoch_freq)

            # Compute Correlation between Human-Engineered and LLM Rewards
            if "gt_reward" in tensorboard_logs and "llm_reward" in tensorboard_logs:
                gt_reward = np.array(tensorboard_logs["gt_reward"])
                llm_reward = np.array(tensorboard_logs["llm_reward"])
                run_stats.reward_correlation = np.corrcoef(gt_reward, llm_reward)[0, 1]

            # Add reward components log to the feedback
            for metric in tensorboard_logs:
                if "/" not in metric:
                    metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                    metric_cur_max = max(tensorboard_logs[metric])
                    metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                    if "consecutive_successes" == metric:
                        run_stats.success = metric_cur_max
                    metric_cur_min = min(tensorboard_logs[metric])
                    if metric != "gt_reward" and metric != "llm_reward":
                        if metric != "consecutive_successes":
                            metric_name = metric 
                        else:
                            metric_name = "task_score"
                        feedback += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                    else:
                        # Provide ground-truth score when success rate not applicable
                        if "consecutive_successes" not in tensorboard_logs:
                            feedback += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
            feedback += self.prompts.code_feedback  
        else:
            # Otherwise, provide execution traceback error feedback
            feedback += self.prompts.execution_error_feedback.format(traceback_msg=traceback_msg)

        feedback += self.prompts.code_output_tip
        run_stats.feedback = feedback
        self.stats_manager.add_run(run_stats)

        return success
    
    def analyze_results(self, iter: int, responses: List[Choice]):
        """
        Analyze RL training results and prepare for the next iteration.
        """
        # Select the best code sample based on the success rate
        best_run_stats = self.stats_manager.get_best_run_within_iteration(iter)
        execute_rate = self.stats_manager.get_execute_rate_within_iteration(iter)
        
        # If all runs failed, it should not reach here
        assert best_run_stats is not None, "All code generation failed in this iteration!"
        
        best_sample_idx = best_run_stats.response_id
        best_feedback = best_run_stats.feedback

        logging.info(f"Iteration {iter}: Execute_rate: {execute_rate*100:.2f}%, Best Success: {best_run_stats.success}, Best Reward Correlation: {best_run_stats.reward_correlation}, Best Code Path: {best_run_stats.code_path}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: LLM Output Content:\n" +  responses[best_sample_idx].message.content + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_feedback + "\n")

        # Plot the success rate
        execute_rates, best_successes, best_reward_correlations, best_code_paths = self.stats_manager.get_best_stat_for_each_iteration(iter)
        
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{self.cfg.env.task}')

        x_axis = np.arange(len(best_successes))

        axs[0].plot(x_axis, np.array(best_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig(self.cfg.paths.summary_figure)
        np.savez(self.cfg.paths.summary_stats, execute_rates=execute_rates, best_successes=best_successes, best_reward_correlations=best_reward_correlations, best_code_paths=best_code_paths)

        self.update_messages(responses[best_sample_idx].message.content, best_feedback)
    
    def update_messages(self, llm_response: str, feedback: str):
        # Update messages for the next iteration
        if len(self.messages) == 2:
            self.messages += [{"role": "assistant", "content": llm_response}]
            self.messages += [{"role": "user", "content": feedback}]
        else:
            assert len(self.messages) == 4
            self.messages[-2] = {"role": "assistant", "content": llm_response}
            self.messages[-1] = {"role": "user", "content": feedback}

        # Save dictionary as JSON file
        with open(self.cfg.paths.message_log, 'w') as file:
            json.dump(self.messages, file, indent=4)

    def run(self):
        # The main iteration loop for improving reward code
        for iter in range(self.cfg.iteration):
            # Generate multiple reward code candidates
            responses = self.generate_candidates(iter)

            # Post-process each generated code and launch RL training
            code_strs: List[str] = []
            rl_runs: List[Optional[subprocess.Popen]] = []
            for response_id in range(self.cfg.sample):
                response_cur = responses[response_id].message.content
                
                code_string = self.post_process(iter, response_id, response_cur)
                code_strs.append(code_string)
                
                success = self.integrate_code_into_env(iter, response_id, code_string)
                if success:
                    rl_run = self.launch_rl_training(iter, response_id)
                    rl_runs.append(rl_run)
                else:
                    rl_runs.append(None)
                
            # Gather RL training results and construct reward reflection
            exec_success = False 
            for response_id, (code_string, rl_run) in enumerate(zip(code_strs, rl_runs)):
                success = self.gather_result(iter, response_id, rl_run)
                if success:
                    exec_success = True
            
            if not exec_success:
                logging.info(f"Iteration {iter}: All code generation failed, requesting LLM to re-generate...")
                feedback = self.stats_manager.get_first_feedback_within_iteration(iter)
                assert feedback is not None
                self.update_messages(responses[0].message.content, feedback)
                continue

            # Analyze results and prepare for the next iteration
            self.analyze_results(iter, responses)
        
        # After all iterations, report the best reward code overall
        best_run_overall = self.stats_manager.get_best_run_overall()
        if best_run_overall is None: 
            logging.info("All iterations of code generation failed, aborting...")
            logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
            exit()
        logging.info(f"Task: {self.cfg.env.task}, Max Training Success {best_run_overall.success}, Correlation {best_run_overall.reward_correlation}, Best Reward Code Path: {best_run_overall.code_path}")
        shutil.copy(best_run_overall.code_path, self.cfg.paths.generated_task_file)