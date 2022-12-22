import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def MetricFunc(label, pred):
	return {'Accuracy': accuracy_score(label, pred), 'AUC': roc_auc_score(label, pred), 'Precision':precision_score(label, pred), 'Recall':recall_score(label, pred), 'F1 Score':f1_score(label, pred)}

max_length = 64
batch_size = 64
epoch = 4
lr = 1e-6
pretrain_path = 'bert-base-uncased'
# pretrain_path = 'roberta-base'

config = AutoConfig.from_pretrained(pretrain_path)
tokenizer = AutoTokenizer.from_pretrained(pretrain_path, config=config)
model = AutoModelForSequenceClassification.from_pretrained(pretrain_path, config=config).cuda()

train_data = json.load(open('data/train_article.json', 'r'))
test_data = json.load(open('data/test_article.json', 'r'))

def tokenize(dataset, shuffle=False):
    # sentences = ["[CLS] " + data['title'] + " [SEP] " + data['content'] + " [SEP]" for data in dataset['articles']]
    sentences = []
    for data in dataset['articles']:
        content = data['content'].split()
        sentences.append("[CLS] " + data['title'] + " [SEP] " + ' '.join(content[:20]) + " [SEP] " + ' '.join(content[-20:]) + " [SEP]")
    # sentences = ["[CLS] " + data['title'] + " [SEP]" for data in dataset['articles']]
    tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]
    print(tokenized_texts[0])

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    out_size = sum([len(sequence) >= max_length for sequence in input_ids])
    print('{} / {} sentences exceeds length limit.'.format(out_size, len(input_ids)))
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")

    attention_masks = [[float(i > 0) for i in sequence] for sequence in input_ids]
    labels = [int(data) for data in dataset['labels']]
    dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels))
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

train_dataloader = tokenize(train_data, shuffle=True)
test_dataloader = tokenize(test_data, shuffle=True)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
params = model.named_parameters()
optimizer = AdamW([
    { 'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': lr, 'ori_lr': lr },
    { 'params': [p for n, p in params if any(nd in n for nd in no_decay)],  'weight_decay': 0.0, 'lr': lr, 'ori_lr': lr }
], correct_bias=False)

def Train():
    model.train()
    tr_loss, tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        b_loss, b_logits = outputs[0], outputs[1]
        b_loss.backward()
        optimizer.step()

        tr_loss += b_loss.item()
        tr_steps += 1
    print("Train loss: {}".format(tr_loss / tr_steps))

def Test():
    model.eval()
    logits, labels = [], []
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logits = outputs[0]
        logits.append(b_logits.cpu())
        labels.append(b_labels.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    labels = torch.cat([_ for _ in labels], dim=0)
    preds = torch.argmax(logits, -1)
    return MetricFunc(labels, preds)

def Main(metric_name):
    best_metric = 0
    for i in range(epoch):
        print('[START] Train epoch {}.'.format(i))
        Train()
        print('[END] Train epoch {}.'.format(i))
        metric = Test()
        print(metric)
        if metric[metric_name] > best_metric:
            best_metric = metric[metric_name]
    print('[BEST ACC: {}]'.format(best_metric))

Main('F1 Score')