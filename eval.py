import pandas as pd
import base64
from openai import OpenAI
import os
from typing import Dict, List, Iterator
from multiprocessing import Pool
from tqdm import tqdm
import re
from datasets import load_dataset
from PIL import Image
import io
import gc

model = "gpt-4o"
api_key = 'YOURTOKEN'
base_url = 'https://api.openai.com/v1'

INSTRUCTIONS = {
    'single_choice': "Please answer this single choice question about the image. The caption of the image is {verified_caption}. The related sentences are {related_sentences}.\n Please answer the question directly. The answer MUST be of the following format: 'Answer: $ANSWER' (without quotes) where $ANSWER is the answer to the problem (the single letter of the correct answer, A, B, C, D, etc.).",
    'multiple_choice': "Please answer this multiple choice question about the image. The caption of the image is {verified_caption}. The related sentences are {related_sentences}.\n Please answer the question directly. The answer MUST be of the following format: 'Answer: $ANSWER' (without quotes) where $ANSWER is the answer to the problem (the letter(s) of the correct answer(s), split by ',').",
    'numerical': "Please answer this numerical question about the image. The unit of the answer is {unit}. The caption of the image is {verified_caption}. The related sentences are {related_sentences}.\n Please answer the question directly.  The answer MUST be of the following format: 'Answer: $ANSWER' (without quotes) where $ANSWER is the answer to the problem (digit number only, without unit or any other text)."
}

def encode_image_from_pil(image: Image.Image) -> str:
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
        image = background
    
    max_size = (1024, 1024)
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image = image.copy()
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=80, optimize=True)
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    buffered.close()
    return encoded

def extract_result(res):
    if not res:
        return ''
    match = re.search(r"(?i)Answer\s*:\s*([^\s\n]+)", res)
    return match.group(1) if match else ''

def normalize_response(response):
    return response.replace("**", "").replace(":", "").replace("$\\boxed{", "").replace("}$", "").replace("\\$", "").replace("$", "").replace("{", "").replace("\\boxed", "")

def check_answer(item: Dict, model_answer: str):
    question_type = item['question_type']
    correct_answer = str(item['answer']) if item['answer'] is not None else ""
    pred_answer = normalize_response(extract_result(model_answer))
    
    if question_type == 'numerical':
        try:
            model_value = float(''.join(c for c in pred_answer if c.isdigit() or c in '.-'))
            correct_value = float(''.join(c for c in correct_answer if c.isdigit() or c in '.-'))
            tolerance = float(item.get('tolerance', 0.0))
            return abs(model_value - correct_value) <= tolerance, pred_answer
        except (ValueError, TypeError):
            return False, pred_answer
    else:
        model_answers = {c for c in pred_answer.upper() if c in 'ABCDEFGHIJKLMN'}
        correct_answers = {c for c in correct_answer.upper() if c in 'ABCDEFGHIJKLMN'}
        return model_answers == correct_answers, pred_answer

def get_model_response(client: OpenAI, image: Image.Image, question: str, question_type: str, 
                      options: str = "", unit: str = "", verified_caption: str = "", related_sentences: str = "") -> str:
    try:
        base64_image = encode_image_from_pil(image)
        
        if question_type == 'numerical':
            prompt = INSTRUCTIONS[question_type].format(unit=unit, verified_caption=verified_caption, related_sentences=related_sentences)
            prompt += f'\nQuestion: {question}'
        else:
            prompt = INSTRUCTIONS[question_type].format(verified_caption=verified_caption, related_sentences=related_sentences)
            prompt += f'\nQuestion: {question}'
            if options:
                prompt += f'\nOptions: {options}'

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=1280,
            temperature=0.0,
            timeout=600
        )
        del base64_image
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""
    finally:
        gc.collect()

def process_single_item(args):
    item, api_key, base_url = args
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        image = item['image']
        model_answer = get_model_response(
            client=client,
            image=image,
            question=item['question_text'],
            question_type=item['question_type'],
            options=str(item.get('options', '')),
            unit=str(item.get('unit', '')),
            verified_caption=str(item.get('verified_caption', '')),
            related_sentences=str(item.get('related_sentences', ''))
        )
        is_correct, pred_answer = check_answer(item, model_answer)
        del image
        return {
            'idx': item['idx'],
            'question_text': item['question_text'],
            'question_type': item['question_type'],
            'answer': item['answer'],
            'options': item.get('options', ''),
            'unit': item.get('unit', ''),
            'verified_caption': item.get('verified_caption', ''),
            'related_sentences': item.get('related_sentences', ''),
            'tolerance': item.get('tolerance', 0.0),
            'model_answer': model_answer,
            'pred_answer': pred_answer,
            'correct_or_not': is_correct
        }
    except Exception as e:
        print(f"Error processing item {item['idx']}: {e}")
        return None
    finally:
        gc.collect()

def load_dataset_streaming(dataset_name: str, split: str = "train") -> Iterator[Dict]:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for idx, item in enumerate(dataset):
        yield {
            'idx': idx,
            'image': item['image'],
            'question_text': item.get('question_text', ''),
            'question_type': item.get('question_type', ''),
            'answer': item.get('answer', ''),
            'options': item.get('options', ''),
            'unit': item.get('unit', ''),
            'verified_caption': item.get('verified_caption', ''),
            'related_sentences': item.get('related_sentences', ''),
            'tolerance': item.get('tolerance', 0.0)
        }

def get_existing_results(output_file: str) -> set:
    if os.path.exists(output_file):
        return set(pd.read_excel(output_file)['idx'].tolist())
    return set()

def main(dataset_name: str = "luojunyu/FinMME", split: str = "train", 
         sample_size: int = None, num_processes: int = 4):
    
    output_file = f'eval_hf_{model.replace("/", "_").replace("-", "_")}.xlsx'
    processed_indices = get_existing_results(output_file)
    
    items_to_process = []
    for item in load_dataset_streaming(dataset_name, split):
        if sample_size and len(items_to_process) >= sample_size:
            break
        if item['idx'] not in processed_indices:
            items_to_process.append(item)
    
    if not items_to_process:
        print("No items to process")
        return
    
    process_args = [(item, api_key, base_url) for item in items_to_process]
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_item, process_args),
            total=len(items_to_process),
            desc="Processing items"
        ))
    
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        new_df = pd.DataFrame(valid_results)
        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_excel(output_file, index=False)
        
        accuracy = combined_df['correct_or_not'].mean()
        print(f"Overall accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='luojunyu/FinMME')
    parser.add_argument('--split', default='train')
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--num_processes', type=int, default=4)
    args = parser.parse_args()
    
    main(args.dataset, args.split, args.sample_size, args.num_processes) 