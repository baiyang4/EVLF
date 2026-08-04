import argparse
import os
import re
import json
from typing import List, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from internvl.model import load_model_and_tokenizer_tsm_sam as load_model_and_tokenizer
from internvl.model import load_model_and_tokenizer_tsm as load_model_and_tokenizer

from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from torchvision import transforms
from internvl.train.dataset import build_transform, dynamic_preprocess
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from internvl.train.tsm_transform import build_transforms as build_transform_tsm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class RGBConverter:
    def __call__(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

def build_picklable_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        RGBConverter(),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def sam_transforms(img_size, ori_h, ori_w):
    transforms = []

    transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))

    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)

def load_sam_img(img_path):
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    image = cv2.imread(img_path)
    image = (image - pixel_mean) / pixel_std
    h, w, _ = image.shape
    # transforms = sam_transforms(1024, h, w)
    transforms = sam_transforms(256, h, w)
    image = transforms(image=image)['image']
    return image

class ImageQuestionDataset(Dataset):
    def __init__(self, questions: List[Dict], image_folder: str, image_size: int):
        self.questions = questions
        self.image_folder = image_folder
        self.transform = build_picklable_transform(input_size=image_size)
        self.transform_sam = build_picklable_transform(input_size=256)
        self.transform_tsm = build_transform_tsm(is_train=False)
        self.image_size = image_size

    def __len__(self):
        return len(self.questions)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image)
        return pixel_values

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = os.path.join(self.image_folder, question['image'])
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)
        tsm_values = self.transform_tsm(image)
        # sam_values = load_sam_img(img_path)
        sam_values = self.transform_sam(image)
        
        return {
            'pixel_values': pixel_values,
            'tsm_values': tsm_values,
            'sam_values': sam_values,
            'question_id': str(question['question_id']),
            'prompt': question['text'],
            # 'label': question.get('label', None),
            'original_idx': idx
        }

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_inference(rank, world_size, args):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    model, tokenizer = load_model_and_tokenizer(args)
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    image_size = model.module.config.force_image_size or model.module.config.vision_config.image_size \
        if isinstance(model, DDP) else model.config.force_image_size or model.config.vision_config.image_size
    
    with open(args.question_file) as f:
        questions = json.load(f)
    
    dataset = ImageQuestionDataset(questions, args.image_folder, image_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    results = []
    
    model.eval()
    if rank == 0:
        progress_bar = tqdm(total=len(dataloader), desc=f'Rank {rank} Processing')
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            tsm_values = batch['tsm_values'].to(device, dtype=torch.bfloat16)
            sam_values = batch['sam_values'].to(device, dtype=torch.bfloat16)
            prompts = batch['prompt']
            
            generation_config = dict(
                do_sample=True if args.temperature > 0 else False,
                num_beams=args.num_beams,
                max_new_tokens=256
            )
            
            # Create num_patches_list for batch
            num_patches_list = [1] * pixel_values.size(0)
            
            # Batch processing
            responses = model.module.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                tsm_values=tsm_values,
                sam_values=sam_values,
                questions=prompts,
                num_patches_list=num_patches_list,
                generation_config=generation_config,
                verbose=False
            ) if isinstance(model, DDP) else model.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                tsm_values=tsm_values,
                sam_values=sam_values,
                questions=prompts,
                num_patches_list=num_patches_list,
                generation_config=generation_config,
                verbose=False
            )
            print(responses[0])
            # Store results with original indices
            for i in range(len(responses)):
                result = {
                    "question_id": batch['question_id'][i],
                    "prompt": batch['prompt'][i],
                    "text": responses[i],
                    # "label": batch['label'][i] if batch['label'][i] is not None else "",
                    "metadata": {},
                    "original_idx": batch['original_idx'][i].item()
                }
                results.append(result)
            
            if rank == 0:
                progress_bar.update(1)
    
    if rank == 0:
        progress_bar.close()
    
    # Save results from this rank
    output_file = f"{args.answers_file}_rank{rank}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    cleanup()

def merge_results(args, world_size, questions):
    """Merge results from all ranks in the correct order"""
    all_results = []
    for rank in range(world_size):
        rank_file = f"{args.answers_file}_rank{rank}.json"
        with open(rank_file) as f:
            results = json.load(f)
            all_results.extend(results)
        os.remove(rank_file)
    
    all_results.sort(key=lambda x: x["original_idx"])

    seen = {}
    unique_results = []
    for item in all_results:
        if item["original_idx"] not in seen:
            seen[item["original_idx"]] = True
            unique_results.append(item)
    all_results = unique_results

    print(len(all_results))
    print(len(questions))
    
    # Verify order matches original questions
    for i, (result, question) in enumerate(zip(all_results, questions)):
        assert str(result["question_id"]) == str(question["question_id"]), \
            f"Result mismatch at position {i}: {result['question_id']} != {question['question_id']}"
    
    # Remove original_idx before saving
    for result in all_results:
        del result["original_idx"]
    
    # Write final results
    with open(args.answers_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--answers-file', type=str, default='./Your_Results')
    parser.add_argument('--question-file', type=str, default='./Your_Results')
    parser.add_argument('--image-folder', type=str, default='/path/to/images')
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='chat')
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--task-type', type=str, default='cls')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    
    with open(args.question_file) as f:
        questions = json.load(f)
    
    mp.spawn(
        run_inference,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )
    
    merge_results(args, args.num_gpus, questions)
    print('Finished inference')