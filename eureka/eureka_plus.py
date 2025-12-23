import json
import logging
import os
import re
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Type, TypeVar

import numpy as np
import matplotlib.pyplot as plt
import yaml
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel

from pool_manager import ImprovePlan, PoolManager, ModuleSpecList, ModuleUsageList
from stats_manager import RunStats, StatsManager
from utils.misc import block_until_training, filter_traceback, set_freest_gpu
from utils.file_utils import load_tensorboard_logs
from utils.extract_task_code import file_to_string, get_function_signature

T = TypeVar('T', bound=BaseModel)
U = TypeVar('U')

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

        # Token usage stats
        self.first_prompt_tokens = 0
        self.total_prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        if cfg.enable_module_pool:
            initial_system_with_pool = self.prompts.initial_system_with_pool.format(
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
    
    
    
    def call_llm_with_retry(
            self,
            sample: int = 1,
            post_process: Callable[[str], U] = lambda x: x,
        ) -> List[U]:
        """
        Call LLM with retry mechanism.
        Return the list of post-processed contents.
        Do not modify messages except temporarily adding retry prompt.
        """
        logging.info(f"Generating {sample} samples with {self.model}")

        total_samples = 0
        chunk_size = sample

        processed_contents: List[U] = []

        first_prompt_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        # Repeatedly call LLM until enough samples are collected
        while total_samples < sample:
            logging.info(f"Total Samples Collected: {total_samples}/{sample}")
            # Attempt to call LLM with retries
            success = False
            for attempt in range(10):
                contents_cur: List[str] = []
                try:
                    # Call LLM
                    logging.info(f"Current Chunk Size: {chunk_size}")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=self.cfg.temperature,
                        n=chunk_size,
                    )

                    # Record token usages
                    prompt_tokens = response.usage.prompt_tokens
                    if attempt == 0 and total_samples == 0:
                        first_prompt_tokens = prompt_tokens
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                    total_tokens += response.usage.total_tokens

                    # Extract contents and validate
                    contents_cur = [choice.message.content for choice in response.choices]
                    if len(contents_cur) != chunk_size:
                        raise ValueError(f"Received {len(contents_cur)} samples, expected {chunk_size} samples!")
                    for i, content in enumerate(contents_cur):
                        if content is None:
                            raise ValueError(f"Received empty content for sample {total_samples + i}!")
                    
                    # logging.info(f"Attempt {attempt+1}: LLM Response: ")
                    # for i, content in enumerate(contents_cur):
                    #     logging.info(f"Sample {total_samples + i}:\n{content}\n")

                    # Post-process contents
                    processed_contents_cur = [post_process(content) for content in contents_cur]
                    
                    # Remove previous retry prompts from messages
                    if attempt > 0:
                        self.messages = self.messages[:-2*attempt]

                    logging.info(f"Successfully generated {chunk_size} samples in attempt {attempt+1}.")
                    success = True
                    processed_contents.extend(processed_contents_cur)
                    total_samples += chunk_size
                    break
                except Exception as e:
                    # Add retry prompts to messages
                    self.messages.append({"role": "system", "content": self.prompts.retry_system})
                    self.messages.append({"role": "user", "content": self.prompts.retry_user.format(
                        invalid_response=str(contents_cur),
                        error_message=str(e),
                    )})
                    if attempt >= 3:
                        chunk_size = max(int(chunk_size / 2), 1)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if not success:
                logging.info("Code terminated due to too many failed attempts!")
                exit()
        
        logging.info(f"First Prompt Tokens (without retry): {first_prompt_tokens}")
        logging.info(f"Total Prompt Tokens (with retry): {total_prompt_tokens}")
        logging.info(f"Total Completion Tokens: {total_completion_tokens}")
        logging.info(f"Total Tokens: {total_tokens}")

        self.first_prompt_tokens += first_prompt_tokens
        self.total_prompt_tokens += total_prompt_tokens
        self.completion_tokens += total_completion_tokens
        self.total_tokens += total_tokens

        return processed_contents[:sample]

    def call_llm_and_parse(
            self,
            user_prompt: str,
            output_model: Type[T],
            sample: int = 1,
            add_llm_response: bool = True,
        ) -> List[T]:
        """
        Call LLM with user prompt and parse the JSON output into the specified Pydantic model.
        Retries up to 10 times if parsing fails.
        Add user prompt with JSON output tip and LLM response to messages.
        Only be called for module pool mode.
        """
        def parse_json(content: str) -> T:
            content = self.extract_json(content)
            parsed_output = output_model.model_validate_json(content)
            return parsed_output

        # Add JSON output tip to the user prompt
        user_prompt_with_tip = user_prompt + self.prompts.json_output_tip.format(schema=output_model.model_json_schema())
        self.messages.append({"role": "user", "content": user_prompt_with_tip})
        parsed_outputs = self.call_llm_with_retry(
            sample=sample,
            post_process=parse_json,
        )
        if add_llm_response:
            assert sample == 1, "Only support adding single LLM response to messages."
            self.messages.append({"role": "assistant", "content": parsed_outputs[0].model_dump_json(indent=4)})
        return parsed_outputs
        
    def call_llm_for_module(self, user_prompt: str) -> Tuple[str, str]:
        """
        Call LLM with user prompt and return the code string and function signature.
        Retries up to 10 times if call fails.
        Do not modify messages except temporarily adding user prompt.
        Return the code string and its function signature.
        Only be called for module pool mode.
        """
        self.messages.append({"role": "user", "content": user_prompt})

        def post_process_for_code_out_class(code_string: str) -> Tuple[str, str]:
            return self.post_process_for_code(code_string, in_class=False)

        processed_contents = self.call_llm_with_retry(
            sample=1,
            post_process=post_process_for_code_out_class,
        )
        self.messages.pop()  # Remove the user prompt after call
        return processed_contents[0]

    def init_pool(self, iter: int):
        """
        Initialize the reward module pool.
        """
        # Design the reward module specifications
        logging.info(f"Iteration {iter}: Designing reward module specifications for the pool...")
        spec_list = self.call_llm_and_parse(
            self.prompts.spec_design.format(
                task_obs_code_string=self.task_obs_code_string,
                task_description=self.task_description,
            ),
            ModuleSpecList,
        )[0]

        # Save module specifications as JSON file
        with open(self.cfg.paths.spec_init_file, 'w') as file:
            file.write(spec_list.model_dump_json(indent=4))

        # Implement each reward module based on the designed specifications
        logging.info(f"Iteration {iter}: Implementing reward modules for the pool...")
        for spec in spec_list.specs:
            logging.info(f"Implementing module: {spec.name}")
            user_prompt = self.prompts.module_implementation.format(specification=spec.model_dump_json(indent=4))
            code_string, signature = self.call_llm_for_module(user_prompt)
            self.pool_manager.add_module(code_string, spec, signature)
        
        logging.info(f"Iteration {iter}: Initialized pool with {len(self.pool_manager.modules)} modules.")

    def improve_pool(self, iter: int):
        # Add the best LLM response from previous iteration to messages
        best_run = self.stats_manager.get_best_run_within_iteration(iter - 1)
        if best_run is None:
            best_run = self.stats_manager.get_first_run_within_iteration(iter - 1)
        
        best_run_id = best_run.response_id
        usage_list = self.pool_manager.get_module_usage_list(iter - 1, best_run_id)
        self.messages.append({"role": "assistant", "content": usage_list.model_dump_json(indent=4)})
        
        # Generate improvement plan for the pool
        improve_plan = self.call_llm_and_parse(
            best_run.signal + self.prompts.pool_improvement_guidance.format(
                task_obs_code_string=self.task_obs_code_string,
                task_description=self.task_description,
                module_pool_details=self.pool_manager.show(view="edit"),
            ),
            ImprovePlan,
        )[0]
        logging.info(f"Iteration {iter}: Improving pool with {len(improve_plan.add_modules)} additions, {len(improve_plan.delete_modules)} deletions, and {len(improve_plan.modify_modules)} modifications.")
        
        # Save improvement plan as JSON file
        with open(self.cfg.paths.improvement_file.format(iter=iter), 'w') as file:
            file.write(improve_plan.model_dump_json(indent=4))
        
        for add_req in improve_plan.add_modules:
            logging.info(f"Adding module: {add_req.spec.name}")
            user_prompt = self.prompts.module_implementation.format(specification=add_req.spec.model_dump_json(indent=4))
            code_string, signature = self.call_llm_for_module(user_prompt)
            self.pool_manager.add_module(code_string, add_req.spec, signature)
        for delete_req in improve_plan.delete_modules:
            logging.info(f"Deleting module: {delete_req.name}")
            self.pool_manager.delete_module(delete_req.name)
        for modify_req in improve_plan.modify_modules:
            logging.info(f"Modifying module: {modify_req.name}")
            module = self.pool_manager.find_module_by_name(modify_req.name)
            user_prompt = self.prompts.module_modification.format(
                spec=module.spec.model_dump_json(indent=4),
                code=module.code,
                description=modify_req.description,
            )
            code_string, signature = self.call_llm_for_module(user_prompt)
            self.pool_manager.modify_module(modify_req.name, code_string, signature)

    def update_pool(self, phase: Phase, iter: int):
        if phase == Phase.BOOTSTRAP:
            self.init_pool(iter)
        elif phase == Phase.ITERATE:
            self.improve_pool(iter)

    def generate_candidates_from_pool(self, phase: Phase, iter: int) -> List[Tuple[str, str]]:
        prompt_template = ""
        if phase == Phase.BOOTSTRAP:
            prompt_template = self.prompts.initial_function_assembly
        elif phase == Phase.ITERATE:
            prompt_template = self.prompts.function_assembly_with_feedback
        
        module_usage_lists = self.call_llm_and_parse(
            prompt_template.format(
                module_pool=self.pool_manager.show(view="assembly"),
            ),
            ModuleUsageList,
            sample=self.cfg.sample,
            add_llm_response=False,
        )
        self.pool_manager.add_module_usage_lists(iter, module_usage_lists)

        function_and_signatures: List[Tuple[str, str]] = []
        for i, module_usage_list in enumerate(module_usage_lists):
            reward_function = self.pool_manager.construct_reward_function(module_usage_list)
            signature, _ = get_function_signature(reward_function, in_class=True)
            function_and_signatures.append((reward_function, signature))
            # Save module usage list as JSON file
            with open(self.cfg.paths.module_usage_file.format(iter=iter, id=i), 'w') as file:
                file.write(module_usage_list.model_dump_json(indent=4))
        
        # Save the current pool to file
        with open(self.cfg.paths.pool_file.format(iter=iter), 'w') as file:
            file.write(self.pool_manager.show(view="debug"))

        # Save messages as JSON file
        with open(self.cfg.paths.message_log, 'w') as file:
            json.dump(self.messages, file, indent=4)

        # Keep only the initial system and the function construction user prompt for next iteration
        self.messages = self.messages[:1] + self.messages[-1:]
        
        return function_and_signatures

    def generate_candidates_from_scratch(self, phase: Phase, iter: int) -> List[Tuple[str, str]]:
        """
        Generate multiple reward code candidates from LLM.
        Return the processed code strings and their function signatures.
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

            # Save messages as JSON file
            with open(self.cfg.paths.message_log, 'w') as file:
                json.dump(self.messages, file, indent=4)

        logging.info(f"Iteration {iter}: Generating {self.cfg.sample} reward function candidates")

        def post_process_for_code_in_class(code_string: str) -> Tuple[str, str]:
            return self.post_process_for_code(code_string, in_class=True)
        
        processed_contents = self.call_llm_with_retry(
            sample=self.cfg.sample,
            post_process=post_process_for_code_in_class,
        )
        
        # Keep only the initial system and first user message for next iteration
        self.messages = self.messages[:2]

        return processed_contents[:self.cfg.sample]
    
    def generate_candidates(self, phase: Phase, iter: int) -> List[Tuple[str, str]]:
        """
        Generate multiple reward code candidates, either from scratch or from the module pool.
        Return the processed code strings and their function signatures.
        """
        if not self.cfg.enable_module_pool:
            return self.generate_candidates_from_scratch(phase, iter)

        # pool-enabled path
        self.update_pool(phase, iter)
        return self.generate_candidates_from_pool(phase, iter)
    
    def extract_json(self, string: str) -> str:
        # Fix the format in case the beginning is truncated
        if string.strip().startswith("{") or string.strip().startswith("["):
            pass
        elif string.strip().startswith("```"):
            pass
        elif string.strip().startswith("json"):
            string = "```" + string.strip()
        elif string.strip().startswith("\""):
            string = "{" + string
        else:
            string = "{\"" + string
        
        # Extract JSON string enclosed in LLM response
        patterns = [
            r'```json(.*?)```',
            r'```(.*?)```',
        ]
        for pattern in patterns:
            extracted_str = re.search(pattern, string, re.DOTALL)
            if extracted_str is not None:
                return extracted_str.group(1).strip()
        return string

    def post_process_for_code(self, code_string: str, in_class: bool) -> Tuple[str, str]:
        """
        Post-process the generated code string and extract function signature.
        Return the processed code string and its function signature.
        """
        # Fix the format in case the beginning is truncated
        if code_string.strip().startswith("py"):
            code_string = "```" + code_string.strip()
        
        # Regex patterns to extract python code enclosed in LLM response
        patterns = [
            r'```python(.*?)```',
            r'```py(.*?)```',
            r'```(.*?)```',
        ]
        for pattern in patterns:
            matching = re.search(pattern, code_string, re.DOTALL)
            if matching is not None:
                code_string = matching.group(1).strip()
                break

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

        # Extract function signature
        try:
            signature, _ = get_function_signature(code_string, in_class)
        except Exception as e:
            logging.info(f"Cannot parse function signature! Error: {e}")
            raise e
        
        return code_string, signature
    
    def integrate_code_into_env(self, iter: int, response_id: int, code_string: str, signature: str):
        """
        Integrate the generated reward code into the environment code and save to file.
        """
        reward_signature = [
            f"self.rew_buf[:], self.rew_dict = {signature}",
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

        # Copy the generated environment code and the reward code to iteration-specific files
        shutil.copy(self.cfg.paths.generated_task_file, self.cfg.paths.generated_task_file_copy.format(iter=iter, id=response_id))
        with open(self.cfg.paths.generated_reward_copy.format(iter=iter, id=response_id), 'w') as file:
            file.writelines(code_string + '\n')

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
    
    def analyze_iter_results(self, iter: int, contents: List[str]):
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
            functions_and_signatures = self.generate_candidates(phase, iter)

            # Integrate each generated code into the environment and launch RL training
            code_strs: List[str] = []
            rl_runs: List[Optional[subprocess.Popen]] = []
            for response_id, (code_string, signature) in enumerate(functions_and_signatures):
                self.integrate_code_into_env(iter, response_id, code_string, signature)
                rl_run = self.launch_rl_training(iter, response_id)
                code_strs.append(code_string)
                rl_runs.append(rl_run)
                
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
            self.analyze_iter_results(iter, code_strs)
        
        # After all iterations, report the best reward code overall
        best_run_overall = self.stats_manager.get_best_run_overall()
        if best_run_overall is None: 
            logging.info("All iterations of code generation failed, aborting...")
            logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
            exit()
        logging.info(f"Task: {self.cfg.env.task}, Max Training Success {best_run_overall.success}, Correlation {best_run_overall.reward_correlation}, Best Reward Code Path: {best_run_overall.code_path}")
        shutil.copy(best_run_overall.code_path, self.cfg.paths.generated_task_file)

        logging.info("Final Token Usage Statistics:")
        logging.info(f"First Prompt Tokens (without retry): {self.first_prompt_tokens}")
        logging.info(f"Total Prompt Tokens (with retry): {self.total_prompt_tokens}")
        logging.info(f"Total Completion Tokens: {self.completion_tokens}")
        logging.info(f"Total Tokens: {self.total_tokens}")