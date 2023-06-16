import torch
from ml_things import plot_dict, plot_confusion_matrix
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, accuracy_score

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification, \
    get_linear_schedule_with_warmup
import data
import gpt2wrapper
# import transformer_block_old
# from torch import nn
# import adapter
# import insert_adapters

torch.cuda.empty_cache()

# global model

nblocks = 3
epochs = 3
batch_size = 4
valid_epoch_cnt = 1
train_epoch_cnt = 1
test_epoch_cnt = 1

adapters = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('device:')
print(device)

# device = torch.device('cpu')

labels_ids = {'neg': 0, 'pos': 1}
n_labels = len(labels_ids)
model_name = 'gpt2'

print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(model_name, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

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
    model = gpt2wrapper.gpt2Wrapper()
else:
    model = init_without_adapters()
# model.to(device)

# Load model to defined device.
# model.to(device)
# print('Model loaded to `%s`'%device)


gpt2_classificaiton_collator = data.Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                               labels_encoder=labels_ids)

# load imdb data
print('Loading training dataset')
train_dataset = data.imdb_dataset(path='aclImdb/train', use_tokenizer=tokenizer)
# print('Created `train_dataset` with %d examples!'%len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=gpt2_classificaiton_collator)
# print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print('Loading validation dataset')
valid_dataset = data.imdb_dataset(path='aclImdb/test', use_tokenizer=tokenizer)
# print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=gpt2_classificaiton_collator)
# print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

# # function for debugging nan output values, haven't figured out how to use it yet
# activation = {}
#
#
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#
#     return hook


def train(dataloader, optimizer_, device_):
    global model, train_epoch_cnt
    predictions_labels = []
    true_labels = []
    total_loss = 0
    batch_cnt = 1

    # --old vars
    total = 0
    correct = 0
    model.train()
    # --

    print("Train Epoch:", train_epoch_cnt)
    for batch in tqdm(dataloader, total=len(dataloader)):
        # if batch_cnt > 5:  #for testing purposes
        #     break
        print('train batch:', batch_cnt)
        labels = batch['labels']
        inputs = batch['input_ids']
        if adapters:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

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
        print('avg_loss, acc of train epoch: ')
        print(avg_epoch_loss)
        print(acc)
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

    #--old vars
    total = 0
    correct = 0
    model.eval()

    if val:
        print("Val Epoch:", valid_epoch_cnt)
    else:
        print("Test Epoch:", test_epoch_cnt)

    for batch in tqdm(dataloader, total=len(dataloader)):
        # if batch_cnt > 5:  #for testing purposes
        #     break
        if val:
            print('valid batch:', batch_cnt)
        else:
            print('test batch:', batch_cnt)
        labels = batch['labels']
        inputs = batch['input_ids']
        if adapters:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # accuracy
            cls = torch.argmax(outputs, dim=1)
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
            print('avg_loss, acc of val epoch: ')
        else:
            print('avg_loss, acc of test epoch: ')
        print(avg_epoch_loss)
        print(acc)
        return acc, avg_epoch_loss
    else:
        avg_epoch_loss = total_loss / len(dataloader)
        return true_labels, predictions_labels, avg_epoch_loss


loss_fn = torch.nn.CrossEntropyLoss()

if adapters:
    learning_rate = 1e-4
else:
    learning_rate = 2e-5
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


for epoch in tqdm(range(epochs)):
    print()
    if adapters:
        print('Training on batches...')
        train_acc, train_loss = train(train_dataloader, optimizer, device)
        print('Validation on batches...')
        val_acc, val_loss = validation(valid_dataloader, device, True)
    else:
        print('Training on batches...')
        train_labels, train_predict, train_loss = train(train_dataloader, optimizer, device)
        train_acc = accuracy_score(train_labels, train_predict)
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(valid_dataloader, device, True)
        val_acc = accuracy_score(valid_labels, valid_predict)


    # Print loss and accuracy values to see how training evolves.
    print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
        train_loss, val_loss, train_acc, val_acc))
    print()

    # Store the loss value for plotting the learning curve.
    all_loss['train_loss'].append(train_loss)
    all_loss['val_loss'].append(val_loss)
    all_acc['train_acc'].append(train_acc)
    all_acc['val_acc'].append(val_acc)

# Plot loss curves.
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# Plot accuracy curves.
plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# Get prediction form model on validation data. This is where you should use
# your test data.
true_labels_test, predictions_labels_test, avg_test_epoch_loss = validation(valid_dataloader, device, False)

# Create the evaluation report.
evaluation_report = classification_report(true_labels_test, predictions_labels_test, labels=list(labels_ids.values()),
                                          target_names=list(labels_ids.keys()))
# Show the evaluation report.
print('evaluation report:')
print(evaluation_report)

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels_test, y_pred=predictions_labels_test,
                      classes=list(labels_ids.keys()), normalize=True,
                      magnify=0.1,
                      )
