from flask import Flask , request
from flask_restful import Resource, Api
from flask_caching import Cache
##########
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
############

app = Flask(__name__)
api = Api(app)

config = {
"DEBUG": True,                    # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 2000 # time out is 2 seconds. yeah, I'll try to decrease it later.
}
cache = Cache(config=config)      # this is for the decorator.

text = {}

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

# @cache.cached(timeout=3000)
def summarize_text_def(text, max_length=0.5, earlyStop=False):
    #TODO: add input support for more Model genereation control.
    #Add num beams control.
    # Add minimum length support, 
    # Add maximum length support,
    # Add limits to all the above functions.
    # Add a feature where you bypass and just output the string as is 
    # If it is less than some number of words.
    preprocess_text = text.strip().replace("\n"," ")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                    num_beams=2,
                                    no_repeat_ngram_size=3,
                                    min_length=int(len(text)/2), #refactor this later
                                    max_length=len(text),
                                    early_stopping=earlyStop)
    text_summarized = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return text_summarized

class summarize_text(Resource):
    def get(self,text_id):
        #TODO: save them inside some proper db I guess?
        text[text_id] = request.form['data']
        return {text_id: summarize_text_def(text[text_id])}

    def put(self,text_id):
        return {text_id: text[text_id]}
api.add_resource(summarize_text, '/<string:text_id>')

class summarize_article_links(Resource):
    pass

if __name__ == '__main__':
    app.config.from_mapping(config)
    app.run(host="0.0.0.0",port="8080")
    # cache = Cache(app)