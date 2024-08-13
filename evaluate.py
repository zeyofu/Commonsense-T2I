import json, time
from tqdm import tqdm
from PIL import Image
import io, os
import random
import argparse
random.seed(42)
from datasets import load_dataset
import collections
from openai import OpenAI
import base64

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def query_gpt4v(model_name, image_path, prompt, retry=10):
    """
    Query the GPT4V model with the image and prompt.
    Feel free to write your own query function for other model."""
    base64_image = encode_image(image_path)
    if model_name == 'GPT4V':
        model_version="gpt-4-vision-preview"
    elif model_name == 'GPT4VT':
        model_version="gpt-4-turbo"
    elif model_name == 'GPT4VO':
        model_version="gpt-4o"
    for r in range(retry):
        try:
            response = client.chat.completions.create(
                model=model_version,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}],
                    }
                ],
                max_tokens=10,
                n=1,
                temperature=0.0,
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
    return 'Failed: Query GPT4V Error'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def eval_mllm(data, t2i_model, evaluator, image_folder, eval_output_path):
    """
    Evaluate the generated images with multimodal LLM such as GPT4V.
    Specifically, for each image, we ask the model to evaluate whether each image fits the description1 and description2.
    Then, we calculate the score based on the pairwise correctness of the two descriptions.
    Input:
    - data: list of dict, including prompts, and the gold description of the correct image
    - t2i_model: str, the model name, e.g., dalle3, flux_schenel, flux_dev, sd_3, sd_xl, LCMs, openjourneyv4, playground25
    - evaluator: str, the model name, e.g. GPT4V
    - image_folder: str, the folder of the generated images
    - eval_output_path: str, the path to save the evaluation results
    Output:
    - None
    """
    if os.path.exists(eval_output_path):
        # Load the existing evaluation results to avoid querying GPT model multiple times
        outputs = json.load(open(eval_output_path, 'r'))
    else:
        outputs = {}
        for i, d in tqdm(enumerate(data)):
            index = i+1
            gpt_answers = {'prompt1_image1_eval1': None, 'prompt1_image1_eval2': None,
                        'prompt1_image2_eval1': None, 'prompt1_image2_eval2': None,
                        'prompt1_image3_eval1': None, 'prompt1_image3_eval2': None, 'prompt1_image4_eval1': None, 'prompt1_image4_eval2': None,
                        'prompt2_image1_eval1': None, 'prompt2_image1_eval2': None, 'prompt2_image2_eval1': None, 'prompt2_image2_eval2': None,
                        'prompt2_image3_eval1': None, 'prompt2_image3_eval2': None, 'prompt2_image4_eval1': None, 'prompt2_image4_eval2': None}
            extracted_answers = []
            prompt1_image_paths = [f'{image_folder}/prompt1_img/original/{str(index).zfill(4)}-{j+1}.jpg' for j in range(4)]
            prompt2_image_paths = [f'{image_folder}/prompt2_img/original/{str(index).zfill(4)}-{j+1}.jpg' for j in range(4)]
            description1 = d["description1"].replace("\n", ", ")
            description2 = d["description2"].replace("\n", ", ")
            degree = 'generally'
            prompt_eval1 = f'Can you tell me if the image {degree} fits the descriptions "{description1}"? If it {degree} fits the descriptions, then return 1, otherwise, return 0. Give me number 1 or 0 only.'
            prompt_eval2 = f'Can you tell me if the image {degree} fits the descriptions "{description2}"? If it {degree} fits the descriptions, then return 1, otherwise, return 0. Give me number 1 or 0 only.'

            for prompt_number, prompt_images in zip(['prompt1', 'prompt2'], [prompt1_image_paths, prompt2_image_paths]):
                for j in range(4):
                    image_path = prompt_images[j]
                    for eval_number, prompt_eval in zip(['description1', 'description2'], [prompt_eval1, prompt_eval2]):
                        try:
                            gpt_answer = query_gpt4v(evaluator, image_path, prompt_eval)
                        except Exception as e:
                            print(e)
                            gpt_answer = 'Failed: Error'
                        gpt_answers[f'{prompt_number}_image{j+1}_{eval_number}'] = gpt_answer
                        try:
                            extracted_answers.append(int(gpt_answer.strip()))
                        except Exception as e:
                            print(e)
                            extracted_answers.append(-1)
            d['prompt1_image_paths'] = prompt1_image_paths
            d['prompt2_image_paths'] = prompt2_image_paths
            d['prompt_eval1'] = prompt_eval1
            d['prompt_eval2'] = prompt_eval2
            d['predictions'] = gpt_answers
            d['prediction_extracted'] = extracted_answers
            d['idx'] = index
            new_d = d
            outputs[index] = new_d
            json.dump(outputs, open(eval_output_path, 'w'), indent=4)
        json.dump(outputs, open(eval_output_path, 'w'), indent=4)

    scores = {int(index): d['prediction_extracted'] for index, d in outputs.items()} 
    score = get_score(scores)
    print(f'{evaluator} eval scores for task {t2i_model} is', round(score/len(scores)*100, 2))


def eval_clip(data, t2i_model, evaluator, image_folder, eval_output_path):
    """
    Evaluate the generated images with CLIP model.
    Specifically, for each image, we calculate the clip similarity with description1 and description2.
    Then, we calculate the score based on the pairwise correctness of the two descriptions.
    Input:
    - data: list of dict, including prompts, and the gold description of the correct image
    - t2i_model: str, the model name, e.g., dalle3, flux_schenel, flux_dev, sd_3, sd_xl, LCMs, openjourneyv4, playground25
    - evaluator: str, the model name, clip
    - image_folder: str, the folder of the generated images
    - eval_output_path: str, the path to save the evaluation results
    Output:
    - None
    """
    def get_clip_score(image_path, evals, model, preprocess, device='cuda'):
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(evals)]).to(device)
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        similarity = similarity.softmax(dim=-1)
        return similarity.cpu().numpy().tolist()[0]
    import clip
    from PIL import Image
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    outputs = {}
    for i, d in tqdm(enumerate(data)):
        index = i+1
        clip_answers = {'prompt1_image1_eval1': None, 'prompt1_image1_eval2': None,
                'prompt1_image2_eval1': None, 'prompt1_image2_eval2': None,
                'prompt1_image3_eval1': None, 'prompt1_image3_eval2': None, 'prompt1_image4_eval1': None, 'prompt1_image4_eval2': None,
                'prompt2_image1_eval1': None, 'prompt2_image1_eval2': None, 'prompt2_image2_eval1': None, 'prompt2_image2_eval2': None,
                'prompt2_image3_eval1': None, 'prompt2_image3_eval2': None, 'prompt2_image4_eval1': None, 'prompt2_image4_eval2': None}
        extracted_answers = []
        prompt1_image_paths = [f'{image_folder}/prompt1_img/original/{str(index).zfill(4)}-{j+1}.jpg' for j in range(4)]
        prompt2_image_paths = [f'{image_folder}/prompt2_img/original/{str(index).zfill(4)}-{j+1}.jpg' for j in range(4)]
        description1 = d["description1"].replace("\n", ", ")
        description2 = d["description2"].replace("\n", ", ")

        for prompt_number, prompt_images in zip(['prompt1', 'prompt2'], [prompt1_image_paths, prompt2_image_paths]):
            for j in range(4):
                image_path = prompt_images[j]
                clip_similarities = get_clip_score(image_path, [description1, description2], model, preprocess)
                clip_answers[f'{prompt_number}_image{j+1}_eval1'] = clip_similarities[0]
                clip_answers[f'{prompt_number}_image{j+1}_eval2'] = clip_similarities[1]
                if clip_similarities[0] > clip_similarities[1]:
                    extracted_answers += [1,0]
                elif clip_similarities[0] < clip_similarities[1]:
                    extracted_answers += [0,1]
                else:
                    extracted_answers += [0,0]
        d['prompt1_image_paths'] = prompt1_image_paths
        d['prompt2_image_paths'] = prompt2_image_paths
        d['prompt_eval1'] = description1
        d['prompt_eval2'] = description2
        d['predictions'] = clip_answers
        d['prediction_extracted'] = extracted_answers
        d['idx'] = index
        outputs[index] = d
        json.dump(outputs, open(eval_output_path, 'w'), indent=4)
    json.dump(outputs, open(eval_output_path, 'w'), indent=4)

    scores = {int(index): d['prediction_extracted'] for index, d in outputs.items()}
    score = get_score(scores)
    print(f'{evaluator} eval scores for task {t2i_model} is', round(score/len(scores)*100, 2))


def tell_supporting_description(score1, score2):
    if score1 == 1 and score2 == 0:
        # image support description1
        return 1
    elif score1 == 0 and score2 == 1:
        # image support description2
        return 2
    elif score1 == 1 and score2 == 1:
        # image support both descriptions, which cannot happen, must be some evaluator deficiency, then random assign one
        return random.choice([1,2])
    else:
        # image fail to support any description
        return 0


def get_score(scores):
    score = 0
    for index, orig_scores in scores.items():
        # Rule: must be pairwise correct to be correct.
        # best case: prompt1_images fit description1, prompt2_images fit description2: score = 1
        # bad case1: prompt1_images fit description2, prompt2_images fit description1: score = 0
        # bad case2: prompt1_images fit neither, prompt2_images fit neither: score = 0
        # bad case3: prompt1_images fit description2, prompt2_images fit description2: score = 0
        support_eval = [tell_supporting_description(orig_scores[i*2], orig_scores[i*2+1]) for i in range(8)]
        # calculate how many correct pairs and take average, a correct pair should be [1,2]
        s = sum([1 for i in range(4) if support_eval[i] == 1 and support_eval[i+4] == 2])/4
        score += s
    return score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=str, default='GPT4VO', help="GPT4VT, GPT4VO, clip, GPT4V")
    parser.add_argument("--t2i_model", type=str, default='dalle', help="dalle3, dalle3_no_revision, flux_schenel, flux_dev, sd_3, sd_xl, LCMs, openjourneyv4, playground25")
    parser.add_argument("--use_negative_prompt", type=int, default=0, help="whether to use negative prompt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Load data and prompts
    raw_data = load_dataset('CommonsenseT2I/CommonsensenT2I')['train']
    data = []
    for i in range(len(raw_data)):
        d = raw_data[i]
        data.append({'prompt1': d['prompt1'], 'prompt2': d['prompt2'], 'description1': d['description1'], 'description2': d['description2']})

    # directory for the generated images
    generated_image_root = './generated_images'
    generated_image_dir = f'{generated_image_root}/{args.t2i_model}{bool(args.use_negative_prompt)*"neg"}_images'

    eval_output_path = f'./evals/{args.evaluator}_eval/{args.t2i_model}{bool(args.use_negative_prompt)*"neg"}.json'
    os.makedirs(f'./evals/{args.evaluator}_eval', exist_ok=True)

    if 'clip' in args.evaluator:
        eval_clip(data, args.t2i_model, args.evaluator, generated_image_dir, eval_output_path)
    else:
        eval_mllm(data, args.t2i_model, args.evaluator, generated_image_dir, eval_output_path)
