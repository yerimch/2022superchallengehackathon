# torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import random
from flask import Flask, jsonify, request
#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
app = Flask(__name__)
bertmodel, vocab = get_pytorch_kobert_model()
device="cpu"

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Setting parameters
max_len = 200
batch_size = 32
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 100
learning_rate =  5e-5


model = torch.load("/home/ubuntu/model/best_model_2epoch_maxlen_200.pt",map_location=torch.device('cpu'))

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def new_softmax(a) :
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)

def get_movie_list(emotion):
    if emotion==0:
        file="anger"
    elif emotion==1:
        file="sadness"
    elif emotion==2:
        file="fear"
    elif emotion==3:
        file="surprise"
    else:
        file="joy"
    file+=".txt"
    data=[]

    with open(f"movie_recommendation/{file}") as f:
        for line in f.readlines():
            data.append(line.strip())
    data=[x.split("\t")[0:2] for x in data]
    index=set()
    while len(index)!=3:
        index.add(random.randint(0,len(data)-1))

    temp=[data[i] for i in index]
    return temp

def get_music_list(emotion):
    if emotion==0 or emotion==3:
        file="Calm"
    elif emotion==1 or emotion==2:
        file="Happy"
    else:
        file="Sad"
    file+=".txt"
    data=[]

    with open(f"music_recommendation/{file}","r",encoding="utf-8") as f:
        for line in f.readlines():
            data.append(line.strip())
    data=[x.split(",")[0:2] for x in data]

    index=set()

    while len(index)!=3:
        index.add(random.randint(0,len(data)-1))

    temp=[data[i] for i in index]
    return temp



@app.route('/predict', methods=['POST'])
def predict():
    predict_sentence=request.get_json()["contents"]

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()
    res=0

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            min_v = min(logits)
            total = 0
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()
            for logit in logits:
                probability.append(np.round(logit, 3))
            res=np.argmax(logits)
    movieList=get_movie_list(res)
    musicList=get_music_list(res)

    return jsonify({
        "emotionType":int(res),
        "movieList" : [
            {
                "movieTitle":str(movieList[0][0]),
                "director":str(movieList[0][1]),
                },
            {
                "movieTitle":str(movieList[1][0]),
                "director":str(movieList[1][1]),
                },
            {
                "movieTitle":str(movieList[2][0]),
                "director":str(movieList[2][1]),
                }],
        "musicList":[
            {
                "musicTitle":str(musicList[0][0]),
                "singer":str(musicList[0][1])
                },
            {
                "musicTitle":str(musicList[1][0]),
                "singer":str(musicList[1][1])
                },
            {
                "musicTitle":str(musicList[2][0]),
                "singer":str(musicList[2][1])
                }]
        })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
