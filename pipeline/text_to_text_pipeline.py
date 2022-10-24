from lib2to3.pgen2 import token
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def preprocess_text(t):
    """ Room for additional preprocessing"""

    return t.strip().replace("\n","")

def run_text_to_text_example():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    text ="""
    The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
     The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
     At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
     "We'll be the comeback kids, all of us," he said. "We want to get our country back."  The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
    """
    summary = summarize(model, tokenizer, device, text)
    print ("\n\nSummarized text: \n",summary)

def summarize(model, tokenizer, device, text, num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=75):

    preprocessed_text = preprocess_text(text)
    t5_prepared_Text = "summarize: "+preprocessed_text
    print ("original text preprocessed: \n", preprocessed_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                    num_beams=num_beams,
                                    no_repeat_ngram_size=2,
                                    min_length=min_length,
                                    max_length=max_length,
                                    early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output
