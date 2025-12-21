import json
import logging
import os
import re
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar

import numpy as np
import matplotlib.pyplot as plt
import yaml
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from pool_manager import ImprovePlan, PoolManager, ModuleSpecList, ModuleUsageList
from stats_manager import RunStats, StatsManager
from utils.misc import block_until_training, filter_traceback, set_freest_gpu
from utils.file_utils import load_tensorboard_logs
from utils.extract_task_code import file_to_string, get_function_signature

T = TypeVar('T', bound=BaseModel)

class Phase(Enum):
    BOOTSTRAP = "bootstrap"
    ITERATE = "iterate"

class Prompts:
    def __init__(self, prompt_dirs: List[str]):
        for prompt_dir in prompt_dirs:
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

        self.prompts = Prompts(cfg.paths.prompt_dirs)
        self.stats_manager = StatsManager()
        self.messages = []
        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
        
        if cfg.enable_module_pool:
            assert self.cfg.sample == 1, "Module pool currently only supports sample=1!"
            initial_system_with_pool = self.prompts.initial_system_with_pool.format(
                task_description=self.task_description,
                task_obs_code_string=self.task_obs_code_string,
                task_reward_signature_string=self.prompts.reward_signature,
            )
            self.messages.append({"role": "system", "content": initial_system_with_pool})
            self.pool_manager = PoolManager()
        else:
            initial_system = self.prompts.initial_system.format(task_reward_signature_string=self.prompts.reward_signature) + self.prompts.code_output_tip
            self.messages.append({"role": "system", "content": initial_system})
        
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
    
    def call_llm_and_parse(self, user_prompt: str, output_model: Type[T]) -> T:
        """
        Call LLM with user prompt and parse the JSON output into the specified Pydantic model.
        Retries up to 10 times if parsing fails.
        Add user prompt with JSON output tip and LLM response to messages.
        """
        # Add JSON output tip to the user prompt
        user_prompt_with_tip = user_prompt + self.prompts.json_output_tip.format(schema=output_model.model_json_schema())
        self.messages.append({"role": "user", "content": user_prompt_with_tip})
        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.cfg.temperature,
                )
                content = response.choices[0].message.content
                logging.info(f"LLM Response Content:\n" + content + "\n")
                content = self.extract_json(content)
                parsed_output = output_model.model_validate_json(content)
                
                # Add the valid response to messages
                self.messages.append({"role": "assistant", "content": content})
                
                return parsed_output
            
            except ValidationError as ve:
                logging.info(f"Attempt {attempt}: JSON parsing error: {ve}")
            except Exception as e:
                logging.info(f"Attempt {attempt}: LLM call error: {e}")
            time.sleep(1)
        raise RuntimeError("Failed to get valid JSON response from LLM after multiple attempts.")

    def call_llm_for_module(self, user_prompt: str, add_tip: bool = True) -> Tuple[str, str]:
        """
        Call LLM with user prompt and return the code string output after post-processing.
        Retries up to 10 times if call fails.
        Do not modify messages except temporarily adding user prompt.
        Return the code string and its function signature.
        """
        # Add module output tip to the user prompt
        if add_tip:
            user_prompt = user_prompt + self.prompts.module_output_tip
        self.messages.append({"role": "user", "content": user_prompt})
        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.cfg.temperature,
                )
                content = response.choices[0].message.content
                
                # Remove the user prompt for next calls
                self.messages.pop()
                
                # Post-process the code string
                content = self.post_process(content)
                signature, _ = get_function_signature(content)
                return content, signature
            
            except ValueError as ve:
                logging.info(f"Attempt {attempt}: Code parsing error: {ve}")
            except Exception as e:
                logging.info(f"Attempt {attempt}: LLM call error: {e}")
            time.sleep(1)
        raise RuntimeError("Failed to get valid code response from LLM after multiple attempts.")

    def init_pool(self, iter: int):
        """
        Initialize the reward module pool.
        """
        # Design the reward module specifications
        logging.info(f"Iteration {iter}: Designing reward module specifications for the pool...")
        spec_list = self.call_llm_and_parse(self.prompts.spec_design, ModuleSpecList)

        # Implement each reward module based on the designed specifications
        logging.info(f"Iteration {iter}: Implementing reward modules for the pool...")
        for spec in spec_list.specs:
            logging.info(f"Implementing module: {spec.name}")
            user_prompt = self.prompts.module_implementation.format(specification=spec.model_dump_json())
            code_string, signature = self.call_llm_for_module(user_prompt)
            self.pool_manager.add_module(code_string, spec, signature)
        
        # Show the initialized pool
        logging.info(f"Iteration {iter}: Initialized pool with {len(self.pool_manager.modules)} modules.")
        logging.info(self.pool_manager.show(view="debug"))

    def improve_pool(self, iter: int):
        # Add the best LLM response from previous iteration to messages
        best_run = self.stats_manager.get_best_run_within_iteration(iter - 1)
        if best_run is None:
            best_run = self.stats_manager.get_first_run_within_iteration(iter - 1)
        
        # Generate improvement plan for the pool
        improve_plan = self.call_llm_and_parse(
            best_run.signal + self.prompts.pool_improvement_guidance.format(
                module_pool_details=self.pool_manager.show(view="edit"),
            ),
            ImprovePlan,
        )
        logging.info(f"Iteration {iter}: Improving pool with {len(improve_plan.add_modules)} additions, {len(improve_plan.delete_modules)} deletions, and {len(improve_plan.modify_modules)} modifications.")
        logging.info(f"Improvement Plan: \n" + improve_plan.model_dump_json())
        
        for add_req in improve_plan.add_modules:
            logging.info(f"Adding module: {add_req.spec.name}")
            user_prompt = self.prompts.module_implementation.format(specification=add_req.spec.model_dump_json())
            code_string, signature = self.call_llm_for_module(user_prompt)
            self.pool_manager.add_module(code_string, add_req.spec, signature)
        for delete_req in improve_plan.delete_modules:
            logging.info(f"Deleting module: {delete_req.name}")
            self.pool_manager.delete_module(delete_req.name)
        for modify_req in improve_plan.modify_modules:
            logging.info(f"Modifying module: {modify_req.name}")
            module = self.pool_manager.find_module_by_name(modify_req.name)
            user_prompt = self.prompts.module_modification.format(
                spec=module.spec.model_dump_json(),
                code=module.code,
                description=modify_req.description,
            )
            code_string, _ = self.call_llm_for_module(user_prompt, add_tip=False)
            self.pool_manager.modify_module(modify_req.name, code_string)
        logging.info(f"Iteration {iter}: Improved pool: \n" + self.pool_manager.show(view="debug"))

    def update_pool(self, phase: Phase, iter: int):
        if phase == Phase.BOOTSTRAP:
            self.init_pool(iter)
        elif phase == Phase.ITERATE:
            self.improve_pool(iter)

    def generate_candidates_from_pool(self, phase: Phase, iter: int) -> List[str]:
        prompt_template = ""
        if phase == Phase.BOOTSTRAP:
            # Remove the specification design messages
            assert len(self.messages) == 3
            self.messages = self.messages[:-2]
            
            prompt_template = self.prompts.initial_function_assembly
        elif phase == Phase.ITERATE:
            prompt_template = self.prompts.function_assembly_with_feedback
        
        module_usage_list = self.call_llm_and_parse(
            prompt_template.format(
                module_pool=self.pool_manager.show(view="assembly"),
            ),
            ModuleUsageList,
        )
        reward_function = self.pool_manager.construct_reward_function(module_usage_list)
        logging.info(f"Iteration {iter}: Generated reward function from pool:\n" + reward_function + "\n")
        
        # Save dictionary as JSON file
        with open(self.cfg.paths.message_log, 'w') as file:
            json.dump(self.messages, file, indent=4)

        if phase == Phase.ITERATE:
            # Remove the messages of the previous iteration
            assert len(self.messages) == 7
            self.messages = self.messages[0] + self.messages[-2:]
        
        return [reward_function]

    def generate_candidates_from_scratch(self, phase: Phase, iter: int) -> List[str]:
        """
        Generate multiple reward code candidates from LLM.
        Return the raw LLM response strings.
        """
        # Add necessary messages based on phase
        if phase == Phase.BOOTSTRAP:
            # Add initial user prompt for BOOTSTRAP phase
            initial_user = self.prompts.initial_user.format(task_obs_code_string=self.task_obs_code_string, task_description=self.task_description)
            self.messages.append({"role": "user", "content": initial_user})
        elif phase == Phase.ITERATE:
            best_run = self.stats_manager.get_best_run_within_iteration(iter - 1)
            guidance = self.prompts.success_guidance
            if best_run is None:
                # If all runs failed in previous iteration, use the first run's content for feedback
                best_run = self.stats_manager.get_first_run_within_iteration(iter - 1)
                guidance = self.prompts.failure_guidance
            guidance += self.prompts.code_output_tip
            
            # Add the best LLM response and feedback to messages
            self.messages += [{"role": "assistant", "content": best_run.content}]
            self.messages += [{"role": "user", "content": best_run.signal + guidance}]

            # Save dictionary as JSON file
            with open(self.cfg.paths.message_log, 'w') as file:
                json.dump(self.messages, file, indent=4)

        contents: List[str] = []
        contents_cur: List[Optional[str]] = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = self.cfg.sample

        logging.info(f"Iteration {iter}: Generating {self.cfg.sample} samples with {self.model}")

        while True:
            if total_samples >= self.cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=self.cfg.temperature,
                        n=chunk_size,
                    )
                    contents_cur = [choice.message.content for choice in response_cur.choices]
                    if len(contents_cur) != chunk_size:
                        raise ValueError(f"Received {len(contents_cur)} samples, expected {chunk_size} samples!")
                    for i, content in enumerate(contents_cur):
                        if content is None:
                            raise ValueError(f"Received empty content for sample {total_samples + i}!")
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

            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens
            contents.extend(contents_cur)
        
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        for id in range(self.cfg.sample):
            logging.info(f"Iteration {iter}: Generated Sample {id}:\n " + contents[id] + "\n")

        # Remove the last round of assistant messages and user prompt for next calls
        if phase == Phase.ITERATE:
            assert len(self.messages) == 4
            self.messages = self.messages[:-2]

        return contents[:self.cfg.sample]
    
    def generate_candidates(self, phase: Phase, iter: int) -> List[str]:
        if not self.cfg.enable_module_pool:
            return self.generate_candidates_from_scratch(phase, iter)

        # pool-enabled path
        self.update_pool(phase, iter)
        return self.generate_candidates_from_pool(phase, iter)
    
    def extract_json(self, string: str) -> str:
        patterns = [
            r'```json(.*?)```',
            r'json(.*?)```',
            r'```(.*?)```',
        ]
        for pattern in patterns:
            extracted_str = re.search(pattern, string, re.DOTALL)
            if extracted_str is not None:
                return extracted_str.group(1).strip()
        return string

    def post_process(self, content_cur: str) -> str:
        """
        Post-process the generated code string, returns the modified code string.
        """
        # Regex patterns to extract python code enclosed in LLM response
        patterns = [
            r'```python(.*?)```',
            r'python(.*?)```',
            r'```py(.*?)```',
            r'```(.*?)```',
        ]
        for pattern in patterns:
            code_string = re.search(pattern, content_cur, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
                break
        code_string = content_cur if not code_string else code_string

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

    def gather_result(self, iter: int, response_id: int, code_str: str, rl_run: Optional[subprocess.Popen]) -> bool:
        """
        Gather RL training results and provide signal.
        Returns success status.
        """
        # initialize RunStats for this run
        code_path = self.cfg.paths.generated_task_file_copy.format(iter=iter, id=response_id)
        run_stats = RunStats(iteration=iter, response_id=response_id, content=code_str, code_path=code_path)
        signal = ""

        rl_filepath = self.cfg.paths.reward_iter_output.format(iter=iter, id=response_id)
        
        if rl_run is None or not os.path.exists(rl_filepath):
            signal += self.prompts.execution_error.format(traceback_msg="Reward code cannot be parsed.")
            run_stats.signal = signal
            self.stats_manager.add_run(run_stats)
            return False

        rl_run.communicate()
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        traceback_msg = filter_traceback(stdout_str)
        if traceback_msg != '':
            # Provide execution traceback error signal
            signal += self.prompts.execution_error.format(traceback_msg=traceback_msg)
            run_stats.signal = signal
            self.stats_manager.add_run(run_stats)
            return False
        
        # If RL execution has no error, provide policy statistics signal
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
        epoch_freq = max(int(max_iterations // 10), 1)
        signal += self.prompts.training_signal.format(epoch_freq=epoch_freq)

        # Compute Correlation between Human-Engineered and LLM Rewards
        if "gt_reward" in tensorboard_logs and "llm_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            llm_reward = np.array(tensorboard_logs["llm_reward"])
            run_stats.reward_correlation = np.corrcoef(gt_reward, llm_reward)[0, 1]

        # Add reward components log to the signal
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
                    signal += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                else:
                    # Provide ground-truth score when success rate not applicable
                    if "consecutive_successes" not in tensorboard_logs:
                        signal += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
        run_stats.signal = signal
        self.stats_manager.add_run(run_stats)

        return True
    
    def analyze_results(self, iter: int, contents: List[str]):
        """
        Analyze RL training results and prepare for the next iteration.
        """
        # Select the best code sample based on the success rate
        best_run_stats = self.stats_manager.get_best_run_within_iteration(iter)
        execute_rate = self.stats_manager.get_execute_rate_within_iteration(iter)
        
        # If all runs failed, it should not reach here
        assert best_run_stats is not None, "All code generation failed in this iteration!"
        
        best_sample_idx = best_run_stats.response_id
        best_signal = best_run_stats.signal

        logging.info(f"Iteration {iter}: Execute_rate: {execute_rate*100:.2f}%, Best Success: {best_run_stats.success}, Best Reward Correlation: {best_run_stats.reward_correlation}, Best Code Path: {best_run_stats.code_path}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: LLM Output Content:\n" +  contents[best_sample_idx] + "\n")
        logging.info(f"Iteration {iter}: Training Signal:\n" + best_signal + "\n")

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

    def run(self):
        # The main iteration loop for improving reward code
        for iter in range(self.cfg.iteration):
            # Generate multiple reward code candidates
            phase = Phase.BOOTSTRAP if iter == 0 else Phase.ITERATE
            contents = self.generate_candidates(phase, iter)

            # Post-process each generated code and launch RL training
            code_strs: List[str] = []
            rl_runs: List[Optional[subprocess.Popen]] = []
            for response_id in range(self.cfg.sample):
                content_cur = contents[response_id]
                
                logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
                code_string = self.post_process(content_cur)
                with open(self.cfg.paths.generated_reward_copy.format(iter=iter, id=response_id), 'w') as file:
                    file.writelines(code_string + '\n')
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
                success = self.gather_result(iter, response_id, code_string, rl_run)
                if success:
                    exec_success = True
            
            if not exec_success:
                logging.info(f"Iteration {iter}: All code generation failed, requesting LLM to re-generate...")
                continue

            # Analyze results and prepare for the next iteration
            self.analyze_results(iter, contents)
        
        # After all iterations, report the best reward code overall
        best_run_overall = self.stats_manager.get_best_run_overall()
        if best_run_overall is None: 
            logging.info("All iterations of code generation failed, aborting...")
            logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
            exit()
        logging.info(f"Task: {self.cfg.env.task}, Max Training Success {best_run_overall.success}, Correlation {best_run_overall.reward_correlation}, Best Reward Code Path: {best_run_overall.code_path}")
        shutil.copy(best_run_overall.code_path, self.cfg.paths.generated_task_file)