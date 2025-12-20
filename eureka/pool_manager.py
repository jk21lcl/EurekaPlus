from dataclasses import dataclass

from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class ModuleSpec(BaseModel):
    name: str = Field(description="The name of the reward module function.")
    inputs: List[str] = Field(description="The list of input variable names for the function.")
    input_types: List[str] = Field(description="The list of input variable types for the function.")
    description: str = Field(description="A brief description of what the reward module does.")

class ModuleSpecList(BaseModel):
    specs: List[ModuleSpec] = Field(description="A list of module specifications.")

class ModuleUsage(BaseModel):
    name: str = Field(description="The name of the module to be used.")
    weight: float = Field(description="The weight assigned to this module when combining outputs.")

class ModuleUsageList(BaseModel):
    usages: List[ModuleUsage] = Field(description="A list of module usages with assigned weights.")

@dataclass
class Module:
    code: str
    spec: ModuleSpec
    signature: str

class PoolManager:
    def __init__(self):
        self.modules: List[Module] = []
    
    def find_module_by_name(self, name: str) -> Optional[Module]:
        for m in self.modules:
            if m.spec.name == name:
                return m
        return None
    
    def add_module(self, code: str, spec: ModuleSpec, signature: str):
        # Skip adding if module with same name already exists
        if self.find_module_by_name(spec.name) is None:
            self.modules.append(Module(code=code, spec=spec, signature=signature))
    
    def show(
        self,
        view: Literal["assembly", "edit", "debug"] = "assembly",
    ) -> str:
        """
        Show the current modules in the pool.

        view:
          - "assembly": spec + signature (for function assembly)
          - "edit": spec + code (for initialization and improvement)
          - "debug": spec + signature + code (for humans)
        """
        blocks = []

        for i, m in enumerate(self.modules):
            block = [f"Module {i}: {m.spec.name}"]

            # --- spec is always shown ---
            block.append("Specification:")
            block.append(f"  Description: {m.spec.description}")
            block.append(f"  Inputs: {', '.join(m.spec.inputs)}")
            block.append(f"  Input Types: {', '.join(m.spec.input_types)}")

            # --- signature ---
            if view in ("assembly", "debug"):
                block.append("Signature:")
                block.append(f"  {m.signature}")

            # --- code ---
            if view in ("edit", "debug"):
                block.append("Implementation:")
                block.append(m.code.strip())

            blocks.append("\n".join(block))

        return "\n\n---\n\n".join(blocks)
    
    def construct_reward_function(
        self,
        usage_list: ModuleUsageList,
    ) -> str:
        """
        Construct the combined reward function code based on the specified module usages.
        The function name should be "compute_reward".
        The input argumments should be the union of all module input arguments.
        The output should be a single float reward value, along with a dict of individual module rewards.
        """
        # Collect all unique inputs
        input_set = set()
        for usage in usage_list.usages:
            module = self.find_module_by_name(usage.name)
            if module is None:
                raise ValueError(f"Module {usage.name} not found in pool.")
            input_set.update(module.spec.inputs)
        input_args = ", ".join(sorted(input_set))
        
        # Construct main reward function code
        lines = []
        lines.append(f"def compute_reward({input_args}) -> Tuple[float, Dict[str, float]]:")
        lines.append("    module_rewards = {}")
        lines.append("    total_reward = 0.0")
        
        for usage in usage_list.usages:
            module = self.find_module_by_name(usage.name)
            # Prepare the call line
            call_args = ", ".join(module.spec.inputs)
            lines.append(f"    reward_{module.spec.name} = {module.spec.name}({call_args})")
            lines.append(f"    module_rewards['{module.spec.name}'] = reward_{module.spec.name}")
            lines.append(f"    total_reward += {usage.weight} * reward_{module.spec.name}")
        
        lines.append("    return total_reward, module_rewards")

        # Construct each module function code that will be included
        for usage in usage_list.usages:
            module = self.find_module_by_name(usage.name)
            lines.append("\n")
            lines.append(module.code.strip())
        
        return "\n".join(lines)