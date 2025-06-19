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
            'knowledge_domain': item.get('knowledge_domain', ''),
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
        new_item = item.copy()
        new_item['idx'] = idx
        yield new_item

def get_existing_results(output_file: str) -> set:
    if os.path.exists(output_file):
        try:
            return set(pd.read_excel(output_file)['idx'].tolist())
        except Exception as e:
            print(f"Could not read existing results from {output_file}: {e}")
            return set()
    return set()

def calculate_finscore(df: pd.DataFrame):
    """Calculates and prints the FinScore, Hallucination Rate, and Final Score."""
    if 'knowledge_domain' not in df.columns or df['knowledge_domain'].isnull().all():
        print("Warning: 'knowledge_domain' column not found or is empty. Cannot calculate FinScore.")
        return

    domain_scores = df.groupby('knowledge_domain')['correct_or_not'].mean()
    domain_normed_scores = domain_scores.mean()
    multi_answer_df = df[(df['question_type'] == 'multiple_choice') & (df['answer'].astype(str).str.len() > 1)]
    hallucination_rate = 0.0
    if not multi_answer_df.empty:
        hallucination_rate = (~multi_answer_df['correct_or_not']).mean()
        
    finscore = domain_normed_scores * (1 - hallucination_rate)
    
    print("-" * 20)
    print("FinMME Benchmark Scores:")
    print(f"FinScore: {finscore:.2f}")
    print("-" * 20)

def main(dataset_name: str = "luojunyu/FinMME", split: str = "train", 
         sample_size: int = None, num_processes: int = 8):
    
    output_file = f'eval_hf_{model.replace("/", "_").replace("-", "_")}.xlsx'
    
    # 1. Load existing results
    try:
        existing_df = pd.read_excel(output_file)
        processed_indices = set(existing_df['idx'].tolist())
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        processed_indices = set()

    # 2. Create a generator for items to process
    items_stream = (item for item in load_dataset_streaming(dataset_name, split) if item['idx'] not in processed_indices)
    if sample_size:
        from itertools import islice
        items_stream = islice(items_stream, sample_size)
    
    process_args_stream = ((item, api_key, base_url) for item in items_stream)
    
    # 3. Process items in parallel
    new_results = []
    with Pool(processes=num_processes) as pool:
        # Estimate total for tqdm progress bar
        FINMME_TOTAL_COUNT = 11099 
        total = sample_size if sample_size else FINMME_TOTAL_COUNT - len(processed_indices)
        
        desc = f"Processing with {num_processes} processes"
        results_iterator = pool.imap_unordered(process_single_item, process_args_stream)
        
        for result in tqdm(results_iterator, total=total, desc=desc):
            if result:
                new_results.append(result)

    # 4. Combine, save, and report results
    if not new_results:
        print("\nNo new items were processed.")
        if existing_df.empty:
            return
        combined_df = existing_df
    else:
        new_df = pd.DataFrame(new_results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).sort_values(by='idx').reset_index(drop=True)

    if not combined_df.empty:
        combined_df.to_excel(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        accuracy = combined_df['correct_or_not'].mean()
        print(f"Overall accuracy: {accuracy:.2%}")
        calculate_finscore(combined_df)
    else:
        print("No results to process or save.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='luojunyu/FinMME')
    parser.add_argument('--split', default='train')
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--num_processes', type=int, default=16)
    args = parser.parse_args()
    
    main(args.dataset, args.split, args.sample_size, args.num_processes) 