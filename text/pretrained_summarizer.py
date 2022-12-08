import torch

from transformers import pipeline


class BartSummarizer:
    FINETUNE_DATASET = ['cnn', 'xsum']
    def __init__(self, dataset='cnn', device=None) -> None:
        if dataset not in self.FINETUNE_DATASET:
            raise ValueError(f'Provided model: {dataset} is not available, available model: {self.FINETUNE_DATASET}')

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model = pipeline(task="summarization", 
                              model=f"facebook/bart-large-{dataset}",
                              device=device)

    
    def __call__(self, long_text, **kwargs):
        with torch.no_grad():
            return self.model(long_text, **kwargs)


class DistilBartSummarizer:
    AVAILABLE_MODEL = ['xsum-12-1', 'xsum-6-6', 'xsum-12-3', 'xsum-9-6',
                       'xsum-12-6', '12-3-cnn', '12-6-cnn', '6-6-cnn']
    def __init__(self, model='xsum-12-3', device=None) -> None:
        if model not in self.AVAILABLE_MODEL:
            raise ValueError(f'Provided model: {model} is not available, available model: {self.AVAILABLE_MODEL}')

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model = pipeline(task="summarization", 
                              model=f"sshleifer/distilbart-{model}",
                              device=device)

    
    def __call__(self, long_text, **kwargs):
        with torch.no_grad():
            return self.model(long_text, **kwargs)


class T5StorySumSummarizer:
    def __init__(self, device=None) -> None:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model = pipeline(task="summarization", 
                              model=f"pszemraj/long-t5-tglobal-base-16384-book-summary",
                              device=device)

    
    def __call__(self, long_text, **kwargs):
        with torch.no_grad():
            return self.model(long_text, **kwargs)