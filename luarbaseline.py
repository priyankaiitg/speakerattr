from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup as bsoup
import sys
from torch.nn.functional import cosine_similarity as cosim

dpath = "/home/priyanka/research/2025/speakerattr/v6opschatlogs/"

chatlog = ["chatlog-113-v6ops-202203211000-00.json", "chatlog-116-v6ops-202303270930-00.json", "chatlog-119-v6ops-202403200930-00.json", "chatlog-114-v6ops-202207261000-00.json", "chatlog-117-v6ops-202307250930-01.json", "chatlog-120-v6ops-202407250930-00.json", "chatlog-115-v6ops-202211110930-00.json", "chatlog-118-v6ops-202311071300-00.json"]

data = []

for file in chatlog:
    with open((dpath + file), "r") as f:
        data.extend(json.load(f))

tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)


#transform input
authorspeech = {}
min_len = 10000
min_batch = 1000
episode_threshold = 5
batch_threshold = 3

for utter in data:
    author = utter['author']
    episode = (bsoup(utter['text'], 'lxml').get_text(separator=' ')).split()

    if author not in authorspeech:
        authorspeech[author] = []
    authorspeech[author].append(episode)

delauthor = []

for author in authorspeech:
    delepisode = []
    for idx, episode in enumerate(authorspeech[author]):
        if len(episode) <= episode_threshold:
            delepisode.append(episode)
    authorspeech[author] = list(filter(lambda x: x not in delepisode, authorspeech[author]))
    if len(authorspeech[author]) <= batch_threshold:
        delauthor.append(author)

authors = {}

for author in authorspeech:
    if author not in delauthor:
        authors[author] = authorspeech[author]

for author in authors:
    if len(authors[author]) < min_batch:
        min_batch = len(authors[author])
    for episode in authors[author]:
        if len(episode) < min_len:
            min_len = len(episode)

batch_size = min_batch
episode_length = min_len

print("batch_size: " + str(batch_size))
print("min_len: " + str(episode_length))

if episode_length <= 0 or batch_size <= 1:
    sys.exit()

authoremb = {}

for author in authors:
    text = []
    split = len(authors[author])/(batch_size-1)
    for i in range(0, batch_size):
        idx = (int)((i*split) - 1)
        text.extend(authors[author][idx][0:episode_length])

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

for author in authorsim:
    print(author + " : ")
    print(authorsim[author])
    print("\n\n")

