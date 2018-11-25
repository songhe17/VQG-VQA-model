import nltk
import json
import pickle
def _prepro(save_path,qa_path,word2idx_path,word2ans_path,feat_path,data_size):
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

    with open(data_path,'rb') as f:
        annotations = json.load(f)
    with open(word2idx_path,'rb') as f:
        word_to_idx = pickle.load(f)#data_prepro.json
    with open(word2ans_path,'rb') as f:
        word_to_ans = pickle.load(f)
    with open(feat_path,'rb') as f:
        features = pickle.load(f)
    data = annotations[:data_size]
    for i in range(data_size):
        sentence = data[i]['question']
        a = data[i]['answer']
        token_sent = nltk.word_tokenize(sentence)
        token_a = nltk.word_tokenize(a)
        embeddings = embed(data[i]['answer'])
        token_sent = [int(word_to_idx[x]) for x in token_sent]
        token_a = [int(word_to_ans[x]) for x in token_a]
        img_name = data[i][img_path]
        img_name = img_name[10:]
        feats = features[img_name]
        data[i].update({'ans_emb':embeddings})
        data[i].update({'token_q':token_sent})
        data[i].update({'token_a'token_a})
        data[i].update({'feats':feats})
    with open(save_path,'rb') as f:
        pickle.dump(data,f)
_prepro('data/prepro.pkl','assets/vqa_raw_train.json','assets/word_to_ans.pkl','assets/word_to_idx.pkl','data/feats.pkl'5000)
