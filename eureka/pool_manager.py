from dataclasses import dataclass

from pydantic import BaseModel, Field
from typing import Dict, Literal, List, Optional

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

class AddModuleRequest(BaseModel):
    spec: ModuleSpec = Field(description="The specification of the reward module function.")
    reasoning: Optional[str] = Field(default=None, description="The reasoning behind adding this module.")

class DeleteModuleRequest(BaseModel):
    name: str = Field(description="The name of the module to be deleted.")
    reasoning: Optional[str] = Field(default=None, description="The reasoning behind deleting this module.")

class ModifyModuleRequest(BaseModel):
    # Specification will remain the same; only code may change
    name: str = Field(description="The name of the module to be modified.")
    description: str = Field(description="The description of the modifications to be made.")
    reasoning: Optional[str] = Field(default=None, description="The reasoning behind modifying this module.")

class ImprovePlan(BaseModel):
    add_modules: List[AddModuleRequest] = Field(default_factory=list, description="List of modules to be added.")
    delete_modules: List[DeleteModuleRequest] = Field(default_factory=list, description="List of modules to be deleted.")
    modify_modules: List[ModifyModuleRequest] = Field(default_factory=list, description="List of modules to be modified.")

@dataclass
class Module:
    code: str
    spec: ModuleSpec
    signature: str

class PoolManager:
    def __init__(self):
        self.modules: List[Module] = []
        self.module_usage_lists: Dict[int, List[ModuleUsageList]] = {}
    
    def find_module_by_name(self, name: str) -> Optional[Module]:
        for m in self.modules:
            if m.spec.name == name:
                return m
        return None
    
    def add_module(self, code: str, spec: ModuleSpec, signature: str):
        # Skip adding if module with same name already exists
        if self.find_module_by_name(spec.name) is None:
            self.modules.append(Module(code=code, spec=spec, signature=signature))
    
    def delete_module(self, name: str):
        self.modules = [m for m in self.modules if m.spec.name != name]
    
    def modify_module(self, name: str, new_code: str):
        module = self.find_module_by_name(name)
        if module is not None:
            module.code = new_code
    
    def add_module_usage_lists(self, iteration: int, usage_lists: List[ModuleUsageList]):
        self.module_usage_lists[iteration] = usage_lists
    
    def get_module_usage_list(self, iteration: int, index: int) -> Optional[ModuleUsageList]:
        if iteration in self.module_usage_lists:
            usage_lists = self.module_usage_lists[iteration]
            if 0 <= index < len(usage_lists):
                return usage_lists[index]
        return None
    
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
        lines.append("@torch.jit.script")
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