from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup as bsoup
import sys
from torch.nn.functional import cosine_similarity as cosim

chatlog = ["chatlog-117-ohai-202307281200-00.json"]

data = []

for file in chatlog:
    with open(file, "r") as f:
        data.extend(json.load(f))

tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)


#transform input
authorspeech = {}
min_len = 10000
min_batch = 1000
episode_threshold = 5

for utter in data:
    author = utter['author']
    episode = (bsoup(utter['text'], 'lxml').get_text(separator=' ')).split()

    if author not in authorspeech:
        authorspeech[author] = []
    authorspeech[author].append(episode)

delauthor = []

for author in authorspeech:
    for idx, episode in enumerate(authorspeech[author]):
        if len(episode) <= episode_threshold:
            authorspeech[author].pop(idx)
    if len(authorspeech[author]) <= 1:
        delauthor.append(author)

for author in delauthor:
    authorspeech.pop(author)

for author in authorspeech:
    if len(authorspeech[author]) < min_batch:
        min_batch = len(authorspeech[author])
    for episode in authorspeech[author]:
        if len(episode) < min_len:
            min_len = len(episode)

batch_size = min_batch
episode_length = min_len

print("batch_size: " + str(batch_size))
print("min_len: " + str(episode_length))

if episode_length <= 0 or batch_size <= 1:
    sys.exit()

authoremb = {}

for author in authorspeech:
    text = []
    split = len(authorspeech[author])/(batch_size-1)
    for i in range(0, batch_size):
        idx = (int)((i*split) - 1)
        text.extend(authorspeech[author][idx][0:episode_length])

# we embed `episodes`, a colletion of documents presumed to come from an author
# NOTE: make sure that `episode_length` consistent across `episode`
# batch_size = 3
# episode_length = 16
# text = [
#    ["Foo"] * episode_length,
#    ["Bar"] * episode_length,
#    ["Zoo"] * episode_length,
#]
#text = [j for i in text for j in i]

    tokenized_text = tokenizer(
        text, 
        max_length=32,
        padding="max_length", 
        truncation=True,
        return_tensors="pt"
    )
    # inputs size: (batch_size, episode_length, max_token_length)
    tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(batch_size, episode_length, -1)
    tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(batch_size, episode_length, -1)
    #print(tokenized_text["input_ids"].size())       # torch.Size([3, 16, 32])
    #print(tokenized_text["attention_mask"].size())  # torch.Size([3, 16, 32])

    authoremb[author] = model(**tokenized_text)
#print(out.size())   # torch.Size([3, 512])

authorsim = {}

for author1 in authoremb:
    for author2 in authoremb:
        dist = (cosim(authoremb[author1],authoremb[author2], dim=-1).mean()).item()
        if author1 not in authorsim:
            authorsim[author1] = {}
        authorsim[author1][author2] = dist

print(authorsim)

