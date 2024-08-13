import json
import random
from datasets import load_dataset

html_head = """

 <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>CommonsenseT2I Visualization</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
        </head>

"""

def generate_html_file(data, model_name, filepath, errors, generated_image_dir):

    # create html content
    html_code = html_head
    html_code += f"""

        <body>
            <div id="top"></div>
            <div>
            <nav class="navbar navbar-expand-lg sticky-top bg-body-tertiary border-body border-bottom mb-3">
                <div class="container-fluid justify-content-sm-center justify-content-md-between px-5 py-3">
                    <a class="navbar-brand h1 text-wrap">Commonsense-T2I Visualization for {model_name}</a>
                    <div class="btn-group" role="group">
                        <a type="button" class="btn btn-outline-primary" href="#top">Top</a>
                    </div>
                </div>
            </nav>
                
            <div class="container text-center">
    """
    count = 1
    # parse content from metadata file
    for i, row_dict in data.items():
        if i in errors:
            error_message = f'<br>\n                        <div class="card-title" style="color:red;">ERROR Detected!</div>'
        else:
            error_message = ''
        prompt1, prompt2, eval1, eval2, likelihood, category = row_dict['prompt1'], row_dict['prompt2'], row_dict['description1'], row_dict['description2'], row_dict['likelihood'], row_dict['category'].title()

        img_1 = f"{generated_image_dir}/prompt1_img/{str(i).zfill(4)}.jpg"
        img_2 = f"{generated_image_dir}/prompt2_img/{str(i).zfill(4)}.jpg"
        # load content into html
        html_code += f"""

        <div class="row">
            <div class="col-sm-12 col-lg-2 mb-3">
                <div class="card">
                    <div class="card-header">{count}</div>
                    <div class="card-body">
                        <div class="card-title">{category}</div>
                        <div class="card-text text-muted">Likelihood: {likelihood}</div>
                        {error_message}
                    </div>
                </div>
            </div>
            <div class="col-sm-6 col-lg-5 mb-3">
                <div class="card">
                    <img src="{img_1}" class="card-img-top" alt="{prompt1}" loading="lazy" />
                    <div class="card-body">
                        <h6 class="card-title" align="left">P1: {prompt1}</h6>
                        <h6 class="card-title" align="left">D1: {eval1}</h6>
                    </div>
                </div>
            </div>
            <div class="col-sm-6 col-lg-5 mb-3">
                <div class="card">
                    <img src="{img_2}" class="card-img-top" alt="{prompt2}" loading="lazy" />
                    <div class="card-body">
                        <h6 class="card-title" align="left">P2: {prompt2}</h6>
                        <h6 class="card-title" align="left">D2: {eval2}</h6>
                    </div>
                </div>
            </div>
        </div>
        
        """
        count += 1
    # Add footer
    html_code += """

            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
        </body>
        </html>

    """

    # Write to a html file
    with open(filepath, 'w') as file:
        file.write(html_code)


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


def get_error_cases(scores, threshold=0.5):
    score = 0
    errors = []
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
        if s <= threshold: # means if fewer than threshold% of images are correct
            errors.append(index)
    return errors


def get_model_errors(evaluator, task):
    if not evaluator:
        return []
    # Load the evaluation output, change the path if needed
    eval_output_path = f'./evals/{evaluator}_eval/{task}.json'
    eval_output = json.load(open(eval_output_path, 'r'))
    scores = {int(index): d['prediction_extracted'] for index, d in eval_output.items()}
    # 0.25 threshold: only show "must incorrect" cases since it's evlauated by model
    errors = get_error_cases(scores, threshold=0.25)
    return errors

if __name__ == "__main__":
    # Define the t2i model images to visualize
    t2i_models = [
                # 'sd_21', 
                # 'sd_xl', 'sd_xlneg',
                # 'sd_3', 'sd_3neg',
                # 'openjourneyv4', 
                # 'playground25', 
                # 'LCMs', 
                # 'flux_schenel', 'flux_dev',
                'dalle3', 
                # 'dalle3_no_revision'
              ]
    # Define the evaluator if want to show t2i model errors
    evaluator = 'GPT4VO'
    # If do not want to show the errors, set evaluator to ''
    # evaluator = ''
    
    # Load the dataset
    raw_data = load_dataset('CommonsenseT2I/CommonsensenT2I')['train']
    data = {}
    for i in range(len(raw_data)):
        d = raw_data[i]
        data[i+1] = {'prompt1': d['prompt1'], 'prompt2': d['prompt2'], 'description1': d['description1'], 'description2': d['description2'], 'likelihood': d['likelihood'], 'category': d['category']}

    # Define the model names to show in the HTML file
    model_names = {
                    'sd_21': "Stable Diffusion 2.1", 
                    'sd_xl': "Stable Diffusion XL", 
                    'sd_xlneg': 'Stable Diffusion XL with negative prompt',
                    'sd_3': 'Stable Diffusion 3 medium',
                    'sd_3neg': 'Stable Diffusion 3m with negative prompt',
                    'openjourneyv4': "OpenJourney v4",
                    'playground25': 'Playground v2.5', 
                    'LCMs': 'Latent Consistency Models (LCMs)', 
                    'flux_schenel': 'Flux Schenel',
                    'flux_dev': 'Flux Dev',
                    'dalle3': 'DALL-E 3', 
                    'dalle3_no_revision': 'DALL-E 3 without prompt revision'
                  }

    for t2i_model in t2i_models:
        use_negative_prompt = True if 'neg' in t2i_model[-3:] else False
        # The path of generated image, change the path if needed
        generated_image_dir = f'./generated_images/{t2i_model}{bool(use_negative_prompt)*"neg"}_images'

        # Get the errors for each model
        model_errors = get_model_errors(evaluator, t2i_model)
        generate_html_file(data, model_name=model_names[t2i_model], filepath=f"visualization_{t2i_model}.html", errors=model_errors, generated_image_dir=generated_image_dir)