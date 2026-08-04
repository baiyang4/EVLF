from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_sequence_utils import BatchFeature
import sys
import os

# The `internvl` package (InternVLSAMModel) and `segment_anything` are shared with the
# EVLF-FM SFT tree; make them importable via PYTHONPATH (see run_scripts/*.sh).
from internvl.model.internvl_chat import InternVLSAMModel as InternVLChatModel
from internvl.model.internvl_chat import InternVLChatConfig
# from torchvision.transforms import functional as F
# from typing import List, Tuple, Optional, Dict
# from torch import Tensor
# import math
from open_r1.eval_metrics.glossary import normalize_word
from open_r1.eval_metrics.utils import split_sentence

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def calculate_f1score(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)
    
    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]
    
    if len(candidate_words) == 0:
        return 0, 0, 0 # "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall


class BYInvernVLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None
        # 1 MedImageInsight + [1 TSM] + 256 SAM. The TSM branch is ablatable via
        # config.use_tsm2, so this must be read off the model, not hardcoded --
        # getting it wrong makes the prompt reserve a different number of
        # <IMG_CONTEXT> slots than the model produces embeddings for, and
        # InternVLSAMModel.generate asserts on that mismatch.
        self.num_extra_encoder_tokens = None

    def get_vlm_key(self):
        return "internvl"
        
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        print(model_id)
        # assert "InternVL" in model_id, f"model_id must contain 'InternVL', but got {model_id}"
        # self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        
        self.model_config = InternVLChatConfig.from_pretrained(model_id)

        # The model class of InternVL when being mapped has been determined by its config
        model_cls = InternVLChatModel
        # InternVL should be inputted with "trust_remote_code=True"
        model_init_kwargs["trust_remote_code"] = True
        # "use_cache" should be removed
        model_init_kwargs.pop("use_cache", None)
        # "flash_attention_2" should be modified to "use_flash_attn" in InternVL
        if "flash_attention_2" in model_init_kwargs.get("attn_implementation", ""):
            model_init_kwargs["use_flash_attn"] = True
            model_init_kwargs.pop("attn_implementation")
        return model_cls

    def post_model_init(self, model, processing_class):
        self.conv_template = model.conv_template if self.conv_template is None else self.conv_template
        self.num_image_token = model.num_image_token if self.num_image_token is None else self.num_image_token
        if self.num_extra_encoder_tokens is None:
            # Same expression as InternVLSAMModel.chat / .forward, kept in sync deliberately.
            self.num_extra_encoder_tokens = 258 if getattr(model, "use_tsm2", True) else 257
            print(f"num_extra_encoder_tokens = {self.num_extra_encoder_tokens} "
                  f"(use_tsm2={getattr(model, 'use_tsm2', True)})")
        img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
    
    def is_embeds_input(self):
        return True

    def get_processing_class(self):
        return AutoProcessor
    
    def get_eos_token_id(self, processing_class):
        eos_token_id = processing_class.convert_tokens_to_ids(self.conv_template.sep.strip())
        return eos_token_id

    def get_vision_modules_keywords(self):
        return ['vision_model', 'tsm_model', 'sam_model', 'tsm_model_2']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'tsm_values', 'sam_values', 'image_flags']
    
    def get_non_generate_params(self):
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('None', 'max_anyres_num')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = []
        for example in inputs:
            template = self.conv_template.copy()
            conversation_list = example["prompt"]
            system_message = extract_system_message(conversation_list)
            if system_message is not None:
                template.system_message = system_message
            
            processed_list = process_conversation_list(conversation_list, system_message)
            for i, processed_item in enumerate(processed_list):
                if i % 2 == 0:
                    template.append_message(template.roles[0], processed_item)
                else:
                    template.append_message(template.roles[1], processed_item)
            if len(processed_list) % 2 == 1:
                template.append_message(template.roles[1], None)
            query = template.get_prompt()
            prompts_text.append(query)
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # Process images
        full_pixel_values = []
        full_tsm_values = []
        full_sam_values = []
        num_patches_list = []
        for img in images:
            pixel_values = self._load_image(img, input_size=self.model_config.vision_config.image_size, max_num=processing_class.max_anyres_num)
            sam_values = self._load_image(img, input_size=256, max_num=processing_class.max_anyres_num)
            tsm_values = self._load_image(img, input_size=480, max_num=processing_class.max_anyres_num)
            full_pixel_values.append(pixel_values)
            full_tsm_values.append(tsm_values)
            full_sam_values.append(sam_values)
            num_patches_list.append(pixel_values.shape[0])

        full_pixel_values = torch.cat(full_pixel_values, dim=0)
        full_tsm_values = torch.cat(full_tsm_values, dim=0)
        full_sam_values = torch.cat(full_sam_values, dim=0)
        
        # Process prompts
        queries = []
        image_idx = 0
        for query in prompts_text:
            while "<image>" in query:
                num_patches = num_patches_list[image_idx]
                n_extra = self.num_extra_encoder_tokens
                if n_extra is None:
                    raise RuntimeError(
                        "num_extra_encoder_tokens is unset -- post_model_init must run before prompt building."
                    )
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.num_image_token + n_extra) * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                image_idx += 1
            queries.append(query)
        assert image_idx == len(num_patches_list)
        
        model_inputs = processing_class(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        model_inputs["pixel_values"] = full_pixel_values
        model_inputs["tsm_values"] = full_tsm_values
        model_inputs["sam_values"] = full_sam_values
        # Only support pure-image data currently (each sample should contain the image)
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)
        
        model_inputs = BatchFeature(data=model_inputs)
        return model_inputs

    def _load_image(self, image: Image.Image, input_size: int=448, max_num:int=12):
        transform = build_transform(input_size=input_size)
        images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @staticmethod
    def get_question_template(task_type: str):

        return (
            # "{Question}\nIdentify the region corresponding to the disease/organ by providing the bounding box coordinates.\n"
            # "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            # "The output answer format should be as follows:\n"
            # "<think>…your step‑by‑step explanation…</think> <answer>[x1, y1, x2, y2]</answer>\n"
            # "Please strictly follow the format."
            "{Question}"  
            "Answer the question based on visual information from the image."  
            "Output the thinking process in <think> </think> and the final answer in <answer> </answer> tags."  
            "The output answer format should be as follows:"  
            "<think>…your step‑by‑step explanation…</think> <answer> final_answer </answer>"  
            "Please strictly follow the format."
        )
        match task_type:
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the InternVL model output matches a specific format."""
        import os
        import re

        allow_answer_first = os.getenv("ALLOW_ANSWER_THINK_ORDER") == "true"
        if allow_answer_first:
            pattern = re.compile(
                r"^(?:<think>.*?</think>\s*)?<answer>.*?</answer>(?:\s*<think>.*?</think>)?$",
                re.DOTALL,
            )
        else:
            pattern = re.compile(
                r"^(?:<think>.*?</think>\s*)?<answer>.*?</answer>$",
                re.DOTALL,
            )
        scores = []
        for completion in completions:
            content = completion[0]["content"]
            scores.append(1.0 if pattern.fullmatch(content) else 0.0)
        return scores

        
    @staticmethod
    def iou_reward_1(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from InternVL model and ground truth bounding box."""
        """Adopt soft iou reward here"""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        # bbox_pattern = r'\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        if iou(bbox, sol) > 0.5:
                            reward = 1.0
                        # reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards




    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from InternVL model and ground truth bounding box."""
        """Adopt soft iou reward here"""
        import re
        import os
        from datetime import datetime
        is_open = False
        open_str = 'none'
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        split_pattern = re.compile(r'[;,]')  #
        def normalize(text):
            # lowercase, strip whitespace and trailing punctuation
            return text.strip().lower().rstrip('.,;!?')


        def reward_f1(pred_set, sol_set):
            inter = pred_set & sol_set
            p = len(inter) / len(pred_set) if pred_set else 1.0
            r = len(inter) / len(sol_set)  if sol_set  else 1.0
            return 2 * p * r / (p + r) if (p + r) else 0.0

        def extract_items(text):
            # split into pieces, normalize each, discard empties
            parts = split_pattern.split(text)
            return { normalize(p) for p in parts if normalize(p) }

        def extract_solution_text(sol):
            if not isinstance(sol, str):
                return sol
            sol_match = re.search(answer_tag_pattern, sol, re.DOTALL)
            return sol_match.group(1).strip() if sol_match else sol

        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            if type(sol) is list:
                try:
                    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                    if content_answer_match:
                        content_answer = content_answer_match.group(1).strip()
                        bbox_match = re.search(bbox_pattern, content_answer)
                        if bbox_match:
                            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]

                            reward =iou(bbox, sol)
                            # reward = iou(bbox, sol)
                except Exception:
                    pass  # Continue to next verification method if this fails
            else:
            # Try symbolic verification first
                try:
                    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                    if content_answer_match:
                        content_answer = content_answer_match.group(1).strip()
                        sol_text = extract_solution_text(sol)
                        if not sol_text.startswith("OPEN:"):
                            
                            pred_set = extract_items(content_answer)
                            sol_set  = extract_items(sol_text)
                            reward = reward_f1(pred_set, sol_set) 
                            # reward = 1.0 if pred_set == sol_set else 0.0


                        else:
                            sol_text = sol_text.removeprefix('OPEN:')
                            is_open = True

                            if sol_text.startswith(("A:", "B:", "C:", "D:")):
                                # print(f"sol: {sol_text}, content_answer: {content_answer}")
                                key_gt, rest_gt = sol_text.split(":", 1)
                                if ":" not in content_answer:
                                    f1_score, precision, recall = calculate_f1score(content_answer, rest_gt)

                                    reward = f1_score
                                    # raise ValueError(f"content_answer missing ':' for MCQ format: {content_answer!r}")
                                else:
                                    key_pred, rest_pred = content_answer.split(":", 1)
                                    f1_score, precision, recall = calculate_f1score(rest_pred, rest_gt)
                                    reward_acc = 1.0 if key_gt == key_pred else 0.0

                                    reward = (reward_acc + recall)/2
                                # print(reward)
                            else:
                                f1_score, precision, recall = calculate_f1score(content_answer, sol_text)
                                open_str = f"f1:{f1_score}, recall: {recall}"
                                # reward = recall
                                reward = f1_score

                            # pred_set = extract_items(content_answer)
                            # sol_set  = extract_items(sol)
                            # reward = 1.0 if pred_set == sol_set else 0.0

                            # reward = iou(bbox, sol)


                except Exception as e:
                    print(f"Error during verification: {e}")
                    if os.getenv("DEBUG_MODE") == "true":
                        log_path = os.getenv("LOG_PATH")
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write("Verification error context:\n")
                            f.write(f"error: {e}\n")
                            f.write(f"sol: {sol}\n")
                            f.write(f"content_answer: {content_answer if 'content_answer' in locals() else None}\n")
                            f.write(f"content: {content}\n")
                    
            rewards.append(reward)
            # open_str = f"f1:{f1_score}, recall: {recall}" if is_open else 'none'
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    
        return rewards

    def keyword_reward(completions, solution, dataset=None, **kwargs):
        import os
        import re
        from datetime import datetime
        answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        kw_tag_pattern = re.compile(r"<kw>(.*?)</kw>", re.DOTALL)

        def normalize(text):
            return text.strip().lower()

        def extract_pred_kw(text):
            return {normalize(m) for m in kw_tag_pattern.findall(text) if normalize(m)}

        def extract_gold_kw(sol):
            if not isinstance(sol, str):
                return set()
            return {normalize(m) for m in kw_tag_pattern.findall(sol) if normalize(m)}

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content, sol in zip(contents, solution):
            gold_set = extract_gold_kw(sol)
            if not gold_set:
                reward = 0.0
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} KW reward skipped -------------\n")
                        f.write(f"Content: {content}\n")
                rewards.append(reward)
                continue

            pred_set = extract_pred_kw(content)
            if not gold_set and not pred_set:
                reward = 1.0
            else:
                inter = pred_set & gold_set
                p = len(inter) / len(pred_set) if pred_set else 0.0
                r = len(inter) / len(gold_set) if gold_set else 0.0
                reward = 2 * p * r / (p + r) if (p + r) else 0.0

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} KW reward: {reward} -------------\n")
                    f.write(f"Pred KW: {sorted(pred_set)}\n")
                    f.write(f"Gold KW: {sorted(gold_set)}\n")
                    f.write(f"Content: {content}\n")

            rewards.append(reward)

        return rewards







def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    if system_message is not None:
        conversation_list = conversation_list[1:]
    processed_list = []
    
    for item in conversation_list:
        role = item["role"]
        content = item["content"]
        
        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    return processed_list

def extract_system_message(conversation_list):
    if conversation_list[0]["role"] == "system":
        if isinstance(conversation_list[0]["content"], list):
            return conversation_list[0]["content"][0]["text"]
        else:
            return conversation_list[0]["content"]
    return None


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
