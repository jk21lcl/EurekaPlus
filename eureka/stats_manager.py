import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DUMMY_FAILURE = -10000.0

@dataclass
class RunStats:
    iteration: int
    response_id: int

    success: float = DUMMY_FAILURE              # consecutive success
    reward_correlation: float = DUMMY_FAILURE   # correlation between LLM reward and gt reward
    code_path: Optional[str] = None             # path to the generated reward code
    feedback: Optional[str] = None              # feedback from human or LLM

    @classmethod
    def get_dummy_failure(cls, iteration: int = -1, response_id: int = -1) -> 'RunStats':
        return cls(iteration=iteration, response_id=response_id)

class IterationStats:
    def __init__(self, iteration: int):
        self.iteration = iteration
        self.runs: List[RunStats] = []
        
        self.best_run: Optional[RunStats] = None
        self.execute_num: int = 0
    
    def add_run(self, run_stats: RunStats):
        """
        Add a RunStats to the iteration statistics.
        Also updates the best run and execution count.
        """
        self.runs.append(run_stats)
        if run_stats.success != DUMMY_FAILURE:
            self.execute_num += 1
            if self.best_run is None or run_stats.success > self.best_run.success:
                self.best_run = run_stats
        
class StatsManager:
    def __init__(self):
        self.iterations: Dict[int, IterationStats] = {}
    
    def add_run(self, run_stats: RunStats):
        """
        Add a RunStats to the appropriate IterationStats.
        """
        iter_num = run_stats.iteration
        if iter_num not in self.iterations:
            self.iterations[iter_num] = IterationStats(iteration=iter_num)
        self.iterations[iter_num].add_run(run_stats)
    
    def get_best_run_within_iteration(self, iteration: int) -> Optional[RunStats]:
        """
        Return the best RunStats within a specific iteration.
        """
        if iteration not in self.iterations:
            warnings.warn(f"Iteration {iteration} not found in stats.", RuntimeWarning)
            return None
        return self.iterations[iteration].best_run
    
    def get_execute_rate_within_iteration(self, iteration: int) -> float:
        """
        Return the execution rate within a specific iteration.
        """
        if iteration not in self.iterations:
            warnings.warn(f"Iteration {iteration} not found in stats.", RuntimeWarning)
            return 0.0
        iteration_stats = self.iterations[iteration]
        if len(iteration_stats.runs) == 0:
            warnings.warn(f"No runs found for iteration {iteration}.", RuntimeWarning)
            return 0.0
        return iteration_stats.execute_num / len(iteration_stats.runs)
    
    def get_best_run_overall(self) -> Optional[RunStats]:
        """
        Return the best RunStats across all iterations.
        """
        valid_best_runs = [iter_stats.best_run for iter_stats in self.iterations.values() if iter_stats.best_run is not None]
        if not valid_best_runs:
            warnings.warn("No valid runs found across all iterations.", RuntimeWarning)
            return None
        return max(valid_best_runs, key=lambda run: run.success)
    
    def get_best_stat_for_each_iteration(self, iteration: int) -> Tuple[List[float], List[float], List[float], List[Optional[str]]]:
        """
        Return lists of execute rates, best successes, best reward correlations,
        and best code paths for each iteration up to the specified iteration.
        """
        execute_rates = [self.get_execute_rate_within_iteration(iter_num) for iter_num in range(iteration + 1)]
        best_runs = []
        for iter_num in range(iteration + 1):
            if iter_num not in self.iterations:
                warnings.warn(f"Iteration {iter_num} not found in stats.", RuntimeWarning)
                continue
            iter_stats = self.iterations[iter_num]
            if iter_stats.best_run is not None:
                best_runs.append(iter_stats.best_run)
            else:
                best_runs.append(RunStats.get_dummy_failure(iteration=iter_num))
        best_successes = [run.success for run in best_runs]
        best_reward_correlations = [run.reward_correlation for run in best_runs]
        best_code_paths = [run.code_path for run in best_runs]
        return execute_rates, best_successes, best_reward_correlations, best_code_paths