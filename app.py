from flask import Flask , request
from flask_restful import Resource, Api
from functools import lru_cache
##########
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
############
from newspaper import Article


#TODO: This text generation is slow. I want to train my own model!
# - Tokenizer sequences longer than 512 dont seem to work well.
# - create a custom tokenizing divider. it divides that huge ass article into smaller ,500~ish sequence length
# - 

app = Flask(__name__)
api = Api(app)

config = {
"DEBUG": True,                    # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 2000 # time out is 2 seconds. yeah, I'll try to decrease it later.
}
# cache = Cache(config=config)      # this is for the decorator.

text = {}

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

@lru_cache(maxsize=400)     #this will save past 400 calls in python3.9 it must be simply cache.
def summarize_text_def(text, max_length=0.5, early_stop=False):
    preprocess_text = text.strip().replace("\n"," ")
    preprocess_text = preprocess_text.strip().replace("\t","  ")

    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                    num_beams=2,
                                    no_repeat_ngram_size=3,
                                    min_length=60, #refactor this later
                                    max_length=150,
                                    early_stopping=early_stop)
    text_summarized = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return text_summarized

class summarize_text(Resource):
    def get(self):
        text = request.form['data']
        if (text.count(" ") < 700):
            return {"Summary": summarize_text_def(text)}
        else: return {"Text is too long, Please understand that the developer is an unemployed student. Help me in & Buy me a Coffee for bigger limits."}

class parse_article_links(Resource):
    @lru_cache(maxsize=400)     #this will save past 400 calls in python3.9 it must be simply cache.
    def get(self):
        url = request.form['data']
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        return summarize_text_def(text)

class parse_reddit_posts(Resource):
    @lru_cache(maxsize=400)     #this will save past 400 calls in python3.9 it must be simply cache.
    def get(self):
        pass

api.add_resource(summarize_text, '/api')
api.add_resource(parse_article_links, '/api/link')
api.add_resource(parse_reddit_posts,'/api/reddit-post')

if __name__ == '__main__':
    app.config.from_mapping(config)
    app.run(host="0.0.0.0",port="8080")
    # cache = Cache(app)