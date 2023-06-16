
import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader


class imdb_dataset(Dataset):
    def __init__(self, path, use_tokenizer):
        self.texts = []
        self.labels = []

        for label in ['pos', 'neg']:
            sentiment_path = os.path.join(path, label)

            files_names = os.listdir(sentiment_path)
            for file_name in tqdm(files_names, desc=f'{label} files'):
                file_path = os.path.join(sentiment_path, file_name)
                content = io.open(file_path, mode='r', encoding='utf-8').read()
                self.texts.append(content)
                self.labels.append(label)

        self.n_examples = len(self.labels)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {'text': self.texts[item],
                'label': self.labels[item]}

class Gpt2ClassificationCollator(object):
    def __init__(self, use_tokenizer, labels_encoder):
        self.use_tokenizer = use_tokenizer
        #self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs.update({'labels': torch.tensor(labels)})

        #del inputs['attention_mask']

        return inputs
