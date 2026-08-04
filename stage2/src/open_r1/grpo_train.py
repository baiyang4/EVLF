# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
# from transformers import Qwen2VLForConditionalGeneration

# `math_verify` was imported here but never used, and it is the only dependency missing
# from the envs that can otherwise run this script (easyr1 / swift).
from open_r1.trainer import GRPOConfig, BYVLMGRPOTrainer
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
# from transformers import TrainingArguments
import yaml
import json
import random
import math

from open_r1.rewards.think_reward import think_reward
# Qwen2.5-VL only; irrelevant to the InternVL path this script runs. It hard-failed at
# import time on transformers >= 4.56, which dropped Qwen2_5_VLVisionFlashAttention2 --
# blocking the whole script over a patch it never applies.
try:
    from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
    monkey_patch_qwen2_5vl_flash_attn()
except ImportError as e:
    warnings.warn(f"Qwen2.5-VL monkey patch unavailable ({e}); skipping. Harmless unless training Qwen2.5-VL.")

    def monkey_patch_qwen2_5vl_forward():
        raise ImportError("Qwen2.5-VL monkey patch is unavailable in this environment.")


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'keyword'.
    """

    reward_funcs: list[str] = field(
        # default_factory=lambda: ["accuracy", "format"],
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'keyword', 'thinking', 'think_align'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    answer_first_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability to place <answer> before <think> in prompts. 0.0 keeps current behavior."
        },
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT_THINK_FIRST = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
SYSTEM_PROMPT_ANSWER_FIRST = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first provides the user with the answer and then gives the reasoning process. The answer and reasoning process are "
    "enclosed within <answer> </answer> and <think> </think> tags, respectively, i.e., "
    "<answer> answer here </answer><think> reasoning process here </think>"
)



class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")

        # Load YAML configuration
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        datasets = yaml_data.get("datasets", [])

        template_dict = {
            "cls": {
                "think_first": (
                    "{Question}\n"
                    "Output the thinking process in <think> </think> and the final answer in <answer> </answer> tags.\n"
                    "The output answer format should be as follows:\n"
                    "<think>...</think> <answer>identified disease</answer>\n"
                    "Please strictly follow the format."
                ),
                "answer_first": (
                    "{Question}\n"
                    "Output the final answer in <answer> </answer> and then the thinking process in <think> </think>.\n"
                    "The output answer format should be as follows:\n"
                    "<answer>identified disease</answer> <think>…your step-by-step explanation…</think>\n"
                    "Please strictly follow the format."
                ),
            },
            # 'cls': (
            #     "{Question}\n"
            #     "Output the final answer in <answer> </answer> tags.\n"
            #     "The output answer format should be as follows:\n"
            #     "<answer>identified disease</answer>\n"
            #     "Please strictly follow the format."
            # ),
            # vqa is answer-only, and stays that way. Both branches are byte-identical
            # (so answer_first_prob is a no-op here) and byte-identical to the eval-time
            # prompt in SFT/eval/eval_batch_sam_r1_mix.py -- that equality is the point,
            # do not "improve" the wording on one side.
            #
            # A <think>-requesting variant was trialled on 2026-07-21 (run
            # InternVL-8b-think_align-v1, 399 steps) because ThinkAlign is structurally 0 without a
            # think block -- 0.0% of vqa rollouts emitted one, against 100% for grounding
            # and 95.3% for cls, so within-group spread was 0 and nothing pushed the
            # policy to start. Training looked like it worked: diag/vqa/iou_reward/mean
            # climbed 0.53 -> 0.80. Eval said otherwise, on every VQA metric:
            #
            #                        GRPO+answer-only   GRPO+think   SFT baseline
            #   VQA-RAD closed           0.9533           0.9183        0.9650
            #   VQA-RAD open recall      0.8755           0.6377        0.9247
            #   Path-VQA closed          0.9131           0.8542        0.9160
            #   Path-VQA open recall     0.2428           0.1110        0.2534
            #   SLAKE closed             0.9246           0.6148        0.9246
            #   SLAKE open recall        0.8918           0.7467        0.8953
            #
            # The training curve was measuring recovery from the damage the think prompt
            # itself caused, against its own degraded start -- it is not comparable across
            # prompts. Only end-task eval settles this. ThinkAlign is therefore inert on vqa
            # (12.5% of the scenario-B mix); cls and grounding keep their think prompts
            # and are unaffected.
            "vqa": {
                "think_first": (
                    "{Question}\n"
                    "The output answer format should be as follows:\n"
                    "<answer>final answer</answer>\n"
                    "Please strictly follow the format."
                ),
                "answer_first": (
                    "{Question}\n"
                    "The output answer format should be as follows:\n"
                    "<answer>final answer</answer>\n"
                    "Please strictly follow the format."
                ),
            },
            "grounding": {
                "think_first": (
                    "{Question}\n"
                    "Identify the region corresponding to the disease/organ by providing the bounding box coordinates.\n"
                    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
                    "The output answer format should be as follows:\n"
                    "<think>...</think> <answer>[x1, y1, x2, y2]</answer>\n"
                    "Please strictly follow the format."
                ),
                "answer_first": (
                    "{Question}\n"
                    "Identify the region corresponding to the disease/organ by providing the bounding box coordinates.\n"
                    "Output the final answer in <answer> </answer> and then the thinking process in <think> </think>.\n"
                    "The output answer format should be as follows:\n"
                    "<answer>[x1, y1, x2, y2]</answer> <think>…your step-by-step explanation…</think>\n"
                    "Please strictly follow the format."
                ),
            },
        }
        self.template_dict = template_dict

        # Iterate datasets in order, matching image_roots and task_types by index
        for idx, data in enumerate(datasets):
            # Select corresponding image_root and task_type


            json_path = data.get("json_path")
            image_base = data.get("image_base")
            # print(f'get image_base = {image_base}')
            task_type = data.get("task_type")
            sampling_strategy = data.get("sampling_strategy", "all")
            sampling_number = None
            # Load JSONL or JSON
            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(json_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                with open(json_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            # Parse sampling strategy
            if ":" in sampling_strategy:
                strategy, num = sampling_strategy.split(":")
                sampling_strategy = strategy
                if "%" in num:
                    pct = int(num.strip("%"))
                    sampling_number = math.ceil(pct * len(cur_data_dict) / 100)
                else:
                    sampling_number = int(num)

            # Apply sampling
            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            print(f"Loaded {len(cur_data_dict)} samples from {json_path}")

            # Annotate each example with image_root, task_type, and question_template
            for ex in cur_data_dict:
                ex["image_base"] = image_base
                ex["task_type"] = task_type
                ex["question_template"] = template_dict[task_type]

            self.list_data_dict.extend(cur_data_dict)

    def __len__(self):
        return len(self.list_data_dict)

    def make_conversation_image(self, formatted):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": formatted},
                    ],
                },
            ],
        }

    def __getitem__(self, i):
        example = self.list_data_dict[i]

        # Load image if present
        image = None
        if 'image' in example:
            img_root = example.get('image_base', self.script_args.image_root)
            image_path = os.path.join(img_root, example['image'])
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, selecting another sample")
                i = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[i]
                img_root = example.get('image_base', self.script_args.image_root)
                image_path = os.path.join(img_root, example['image'])
            image = Image.open(image_path).convert("RGB")

        # Build prompt based on task_type
        tpl_entry = example.get("question_template", None)
        use_answer_first = random.random() < max(self.script_args.answer_first_prob, 0.0)
        if isinstance(tpl_entry, dict):
            tpl = tpl_entry["answer_first"] if use_answer_first else tpl_entry["think_first"]
        else:
            tpl = tpl_entry
        formatted = tpl.format(Question=example.get("problem", ""))

        if 'image' not in example:
            # Text-only prompt
            system_prompt = SYSTEM_PROMPT_ANSWER_FIRST if use_answer_first else SYSTEM_PROMPT_THINK_FIRST
            prompt =  [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example.get("problem", "")},
            ]
        else:
            # Multimodal prompt (image + text)
            prompt = self.make_conversation_image(formatted)
           
        # print(formatted)
        # print(prompt)
        # COT fields: support old format (cot) and new dual format (reasoning_content_gpt/gemini)
        cot_gpt = example.get('reasoning_content_gpt', example.get('cot', ''))
        cot_gemini = example.get('reasoning_content_gemini', '')
        return {
            'image': image,
            'problem': example.get('problem'),
            'solution': example.get('solution', example.get('content', '')),
            'dataset': example.get('dataset'),
            'task_type': example.get('task_type'),  # logging only: per-task GRPO diagnostics
            'cot_gpt': cot_gpt,
            'cot_gemini': cot_gemini,
            'prompt': prompt['prompt'],
        }
    


def get_vlm_module(model_name_or_path):
    # EVLF-FM Stage-2 is InternVL-only.
    if "internvl" in model_name_or_path.lower():
        return BYInvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    if script_args.answer_first_prob > 0:
        os.environ["ALLOW_ANSWER_THINK_ORDER"] = "true"

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.iou_reward,
        "keyword": vlm_module_cls.keyword_reward,
        "think": think_reward,
        # "format": vlm_module_cls.format_reward_rec,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = BYVLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
