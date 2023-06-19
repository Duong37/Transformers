import torch
from sklearn.model_selection import train_test_split
# from ml_things import plot_dict, plot_confusion_matrix
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
import wandb

from sklearn.metrics import classification_report, accuracy_score

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification, \
    get_linear_schedule_with_warmup
import data
# import gpt2wrapper
from torch import nn
import adapter
import pandas as pd

# torch.cuda.empty_cache()

# global model
# 1e-4 for adapter tuning
# 2e-5 for without adapters
wandb.init(name="small_test2",
           project="Adapter-based tuning of GPT-2",
           entity="d-vuhai",
           config={"learning_rate": 1e-4, "batch_size": 32},
           )

config = wandb.config
config.learning_rate = 1e-4

'''
check batch_size
freeze_weights
insert_at
wandb init name
learning rate
disable testing in train/valid loops
'''

nblocks = 3
epochs = 3
batch_size = 32
valid_epoch_cnt = 1
train_epoch_cnt = 1
test_epoch_cnt = 1

freeze_weights = False  # freeze gpt2 weights
adapters = True  # use wrapper
insert_at = 'none'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device:')
print(device)

# device = torch.device('cpu')

labels_ids = {'negative': 0, 'positive': 1}
n_labels = len(labels_ids)
model_name = 'gpt2'

print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(model_name, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

class NoParam(nn.Module):
    """
    Wraps a module, stopping parameters from being registered
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = [mod]

    def cuda(self):
        self.mod[0].cuda()

    def forward(self, x, *args, **kwargs):
        return self.mod[0](x, *args, **kwargs)


class gpt2Wrapper(nn.Module):
    def __init__(self, iblocks=3, model_name='gpt2', dropout=0.0, csize=None):
        super().__init__()

        self.labels_ids = {'neg': 0, 'pos': 1}
        self.n_labels = len(self.labels_ids)

        self.model_config = GPT2Config.from_pretrained(model_name, num_labels=self.n_labels)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model = GPT2LMHeadModel.from_pretrained(model_name, config = self.model_config)
        self.model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                                   config=self.model_config)

        if freeze_weights:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        emb = self.model.config.n_embd
        self.ctx = self.model.config.n_ctx
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.container = [None]

        self.iblocks = nn.ModuleList([
            adapter.AdapterBlock(emb, 8, True, 4, dropout, csize=csize,
                                 cond=self.container) for _ in range(iblocks)
        ])

        h = self.model.transformer.h  # the main stack of transformer blocks

        if insert_at == 'start':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i)
                block = self.iblocks[i]
                h.insert(i, block)
        elif insert_at == 'middle':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i + 5)
                block = self.iblocks[i]
                h.insert(i + 5, block)
        elif insert_at == 'end':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i + 10)
                block = self.iblocks[i]
                h.insert(i + 10, block)
        elif insert_at == 'everywhere':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i * 6)
                block = self.iblocks[i]
                h.insert(i * 6, block)
        else:
            pass

        # nb = len(self.model.transformer.h) # number of GPT2 blocks
        # print('len nb: ')
        # print(nb)
        # per = nb // iblocks

        # h = self.model.transformer.h  # the main stack of transformer blocks
        # for i in range(iblocks - 1, -1, -1):
        #      print('inserting block at', i * per)
        #      block = self.iblocks[i]
        #      h.insert(i * per, block)
        # h.insert(len(h), self.iblocks[-1])
        #

        print('len h after adding adapters:')
        print(len(h))

        self.register_buffer(name='head_mask', tensor=torch.ones(len(h), self.model.config.n_head))

        self.model = NoParam(self.model)

        # Out own language model head
        # self.headbias = nn.Parameter(torch.zeros(self.tokenizer.vocab_size))  # to token probabilities

    # def forward(self, x, cond=None, layer_past = None, input_ids = None, attention_mask = None, labels = None):
    def forward(self, x, cond=None):
        # b = x.size(0)

        if cond is not None:
            self.container[0] = cond

        x = self.model(x, head_mask=self.head_mask)[0]
        # x =  0.0 * cond.view(b, -1).sum(dim=1) #hack
        # x = x + self.headbias

        return x


def init_without_adapters():
    print('Loading model...')
    # model = GPT2LMHeadModel.from_pretrained(model_name, config=model_config)
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)
    # model.eval()
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    return model


# initialize with wrapper
if adapters:
    model = gpt2Wrapper()
else:
    model = init_without_adapters()
# model.to(device)

# Load model to defined device.
# model.to(device)
# print('Model loaded to `%s`'%device)


gpt2_classificaiton_collator = data.Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                               labels_encoder=labels_ids)

# data_path = '/var/scratch/dvi230/Transformers/IMDB Dataset.csv'
data_path = 'IMDB Dataset.csv'
df = pd.read_csv(data_path)

# Select 4500 positive and 4500 negative samples with labels
positive_samples = df[df['sentiment'] == 'positive'].sample(n=3600, random_state=42)
negative_samples = df[df['sentiment'] == 'negative'].sample(n=3600, random_state=42)

# Combine positive and negative samples
combined_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)

# Split into train, validation, and test sets
train_samples, temp_samples = train_test_split(combined_samples, test_size=2 / 3,
                                               stratify=combined_samples['sentiment'], random_state=42)
valid_samples, test_samples = train_test_split(temp_samples, test_size=0.5, stratify=temp_samples['sentiment'],
                                               random_state=42)

# Get train_x, train_y, valid_x, valid_y, test_x, test_y
train_x = train_samples['review'].tolist()
train_y = train_samples['sentiment'].tolist()
valid_x = valid_samples['review'].tolist()
valid_y = valid_samples['sentiment'].tolist()
test_x = test_samples['review'].tolist()
test_y = test_samples['sentiment'].tolist()

train_dataset = list(zip(train_x, train_y))
valid_dataset = list(zip(valid_x, valid_y))
test_dataset = list(zip(test_x, test_y))

# train_dataset = data.imdb_dataset(path='/var/scratch/dvi230/Transformers/aclImdb/train', use_tokenizer=tokenizer)
# train_dataset = data.imdb_dataset(path='aclImdb/train', use_tokenizer=tokenizer)

print('Created `train_dataset` with %d examples!'%len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=gpt2_classificaiton_collator)

# print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print('Loading validation dataset')
# valid_dataset = data.imdb_dataset(path='/var/scratch/dvi230/Transformers/aclImdb/test', use_tokenizer=tokenizer)
#valid_dataset = data.imdb_dataset(path='aclImdb/test', use_tokenizer=tokenizer)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=gpt2_classificaiton_collator)


print('Created `test_dataset` with %d examples!'%len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=gpt2_classificaiton_collator)


def train(dataloader, optimizer_, device_):
    global model, train_epoch_cnt
    predictions_labels = []
    true_labels = []
    total_loss = 0
    batch_cnt = 1

    total = 0
    correct = 0
    if not freeze_weights:
        model.train()

    wandb.watch(model, loss_fn, log='all', log_freq=10)

    print("Train Epoch:", train_epoch_cnt)
    for batch in tqdm(dataloader, total=len(dataloader)):
        # if batch_cnt > 2:  # for testing purposes
        #     break
        if batch_cnt == 1 or batch_cnt % 100 == 0 or batch_cnt == len(dataloader):
            print('train batch:', batch_cnt)
        labels = batch['labels']
        inputs = batch['input_ids']
        if adapters:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #wandb.log({'grad_clip': clip})
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            wandb.log({'epoch': train_epoch_cnt, 'train_loss': loss.item()})
            # accuracy
            cls = torch.argmax(outputs, dim=1)
            correct += sum(cls == labels).item()
            total += inputs.size(0)
        else:
            true_labels += batch['labels'].cpu().numpy().flatten().tolist()
            batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

            model.zero_grad()

            outputs = model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #wandb.log({'grad_clip': clip})
            optimizer.step()
            scheduler.step()

            logits = logits.detach().cpu().numpy()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()

        batch_cnt += 1

    print('End of Train Epoch:', train_epoch_cnt)
    train_epoch_cnt += 1

    if adapters:
        avg_epoch_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        print('avg_loss, acc of train epoch:')
        print(avg_epoch_loss)
        print(acc)
        wandb.log({'epoch': train_epoch_cnt, 'train_acc': acc})
        return acc, avg_epoch_loss
    else:
        avg_epoch_loss = total_loss / len(dataloader)
        return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, device_, val):
    global model, test_epoch_cnt, valid_epoch_cnt
    predictions_labels = []
    true_labels = []
    total_loss = 0
    batch_cnt = 1

    # --old vars
    total = 0
    correct = 0
    model.eval()

    wandb.watch(model, loss_fn, log='all', log_freq=10)
    if val:
        print("Val Epoch:", valid_epoch_cnt)
    else:
        print("Test Epoch:", test_epoch_cnt)

    for batch in tqdm(dataloader, total=len(dataloader)):
        # if batch_cnt > 2:  # for testing purposes
        #     break
        if val:
            if batch_cnt == 1 or batch_cnt % 100 == 0 or batch_cnt == len(dataloader) - 1:
                print('valid batch:', batch_cnt)
        else:
            if batch_cnt == 1 or batch_cnt % 100 == 0 or batch_cnt == len(dataloader) - 1:
                print('test batch:', batch_cnt)
        labels = batch['labels']
        inputs = batch['input_ids']
        if adapters:
            with torch.no_grad():
                true_labels += batch['labels'].numpy().flatten().tolist()  # test set
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                wandb.log({'epoch': valid_epoch_cnt, 'val_loss': loss.item()})

                # accuracy
                cls = torch.argmax(outputs, dim=1)
                predictions_labels += cls  # test set
                correct += sum(cls == labels).item()
                total += inputs.size(0)
        else:
            true_labels += batch['labels'].numpy().flatten().tolist()
            batch = {k: v.type(torch.long).to(device_) for k, a, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                total_loss += loss.item()

                predict_content = logits.argmax(axis=-1).flatten().tolist()
                predictions_labels += predict_content

        batch_cnt += 1

    if val:
        print('End of Val Epoch:', valid_epoch_cnt)
        valid_epoch_cnt += 1
    else:
        print('End of Test Epoch:', test_epoch_cnt)
        test_epoch_cnt += 1

    if adapters:
        avg_epoch_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        if val:
            print('avg_loss, acc of val epoch:')
            print(avg_epoch_loss)
            print(acc)
            wandb.log({'epoch': valid_epoch_cnt, 'val_acc': acc})
            return acc, avg_epoch_loss
        else:
            print('avg_loss, acc of test epoch:')  # test set
            return true_labels, predictions_labels, avg_epoch_loss

    else:
        avg_epoch_loss = total_loss / len(dataloader)
        return true_labels, predictions_labels, avg_epoch_loss


loss_fn = torch.nn.CrossEntropyLoss()

if insert_at =='none':
    learning_rate = 2e-5
else:
    learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8 # default is 1e-8.
#                   )

total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss': [], 'val_loss': []}
all_acc = {'train_acc': [], 'val_acc': []}
all_values = []
eval_rep_values = []

#
# true_labels_test, predictions_labels_test, avg_test_epoch_loss = [],[],0

for epoch in tqdm(range(epochs)):
    print()
    if adapters:
        print('Training on batches...')
        train_acc, train_loss = train(train_dataloader, optimizer, device)
        print('Validation on batches...')
        val_acc, val_loss = validation(valid_dataloader, device, True)
        print('Testing on batches...')


    else:
        print('Training on batches...')
        train_labels, train_predict, train_loss = train(train_dataloader, optimizer, device)
        train_acc = accuracy_score(train_labels, train_predict)
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(valid_dataloader, device, True)
        val_acc = accuracy_score(valid_labels, valid_predict)
        print('Testing on batches...')

    # Print loss and accuracy values to see how training evolves.
    print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
        train_loss, val_loss, train_acc, val_acc))
    print()

    # Store the loss value for plotting the learning curve.
    all_loss['train_loss'].append(train_loss)
    all_loss['val_loss'].append(val_loss)
    all_acc['train_acc'].append(train_acc)
    all_acc['val_acc'].append(val_acc)
    all_values.append(train_loss)
    all_values.append(val_loss)
    all_values.append(train_acc)
    all_values.append(val_acc)

# Get prediction form model on validation data. This is where you should use
# your test data.
true_labels_test, predictions_labels_test, avg_test_epoch_loss = validation(test_dataloader, device, False)

eval_rep_values.append(true_labels_test)
eval_rep_values.append(predictions_labels_test)

print('all values:')
print(all_values)
print('eval rep values:')
print(eval_rep_values)

wandb.finish()

# Plot loss curves.
# plot1 = plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
#
# # Plot accuracy curves.
# plot2 = plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
#
# # Create the evaluation report.
# evaluation_report = classification_report(true_labels_test, predictions_labels_test, labels=list(labels_ids.values()),
#                                           target_names=list(labels_ids.keys()))
# Show the evaluation report.
# print('evaluation report:')
# print(evaluation_report)

# Plot confusion matrix.
# plot_confusion_matrix(y_true=true_labels_test, y_pred=predictions_labels_test,
#                       classes=list(labels_ids.keys()), normalize=True,
#                       magnify=0.1,
#                       )
