#import pandas as pd
import numpy as np
import vertexai
from vertexai.language_models import TextGenerationModel
import time
from flask import Flask, request, jsonify

project_id = "ga-api-283617"
location = "us-central1"
vertexai.init(project=project_id, location=location)

model = TextGenerationModel.from_pretrained("text-bison@002")

app = Flask(__name__)

@app.route('/submitShoePreferences', methods = ['POST'])
def process_shoe_preferences():
    data = request.json
    shoe_type = data['shoe_type']
    season = data['season']
    feel = data['feel']
    tonality = data['tonality']
    promptype = data['promptype']

    parameters = {
    "temperature": 0.9,
    "max_output_tokens": 1024,
    "top_p": 0.9,
    "top_k": 40
    }

    initial_charac_prmpt= '''You are a marketing expert and a great skilled email writer.
    You have been assigned a task to generate marketing email subject line for shoe products based on below characteristics  provided.
    The Starting line for subject line should be catchy,appealing and also relevant to the characterisitcs.
    The choice of words in subject line should feel {feel} and tonality should be  usually {tonality}.'''

    initial_prompt =  initial_charac_prmpt.format(feel = feel,tonality = tonality)

    discntng_prmpt= '''
    The marketing content is to be generated for {shoe_typ} shoe type alluring the customer with discount.
    Strictly do not mention any dollar value and always provide a [$__] placeholder to mention the dicsount amount.
    Generate atmax 4 subject lines.
    subject line :
    '''
    disc_prompt=  initial_charac_prmpt.format(feel = feel,tonality = tonality)+discntng_prmpt.format(shoe_typ = shoe_type)

    pricing_prmpt= '''
    The marketing content is to be generated for {shoe_typ} shoe type alluring the customer with great price.
    Strictly do not mention any dollar value and always provide a [$__] placeholder to mention the price amount.
    Generate atmax 4 subject lines.
    subject line :
    '''
    price_prompt = initial_charac_prmpt.format(feel = feel,tonality = tonality)+pricing_prmpt.format(shoe_typ = shoe_type)

    seasonal_prmpt = '''
    The marketing content is to be generated for {shoe_typ} shoe type alluring the customer with great price and discount for {seasn} footwear.
    Strictly do not mention any dollar amount value and always provide a [$__] placeholder to mention the price amount at the end.
    Generate atmax 4 subject lines.
    subject line :
    '''
    seasonal_prompt = initial_charac_prmpt.format(feel = feel,tonality = tonality)+seasonal_prmpt.format(shoe_typ = shoe_type,seasn = season)

    generated_subject_lines = []

    if promptype == 'discount':
        generated_subject_lines = model.predict(price_prompt,**parameters)
    elif promptype == 'price':
        generated_subject_lines = model.predict(disc_prompt,**parameters)
    elif promptype == 'season':
        generated_subject_lines = model.predict(seasonal_prompt,**parameters)
    else:
        generated_subject_lines = model.predict(initial_charac_prmpt,**parameters)

    return jsonify({"message": "Preferences received successfully", "subject_lines": generated_subject_lines })

if __name__ == "__main__":
    app.run(host = "localhost", port = 5000)