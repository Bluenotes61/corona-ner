import json, re
from os.path import join

def get_silver_data(train_split = 0.7, max_docs = 0, collected_docs = False):
    fname = 'corona_silver.json'
    if collected_docs:
        fname = 'corona_silver_collected.json'
    docs = json.loads(open(join('data', fname), encoding='utf-8').read())
  
    data = []
    labels = []
    for doc in docs:
        for paragraph in doc['paragraphs']:
            entities = []
            for entity in paragraph['entities']:
                entities.append((entity['start'], entity['end'], entity['label']))
                labels.append(entity['label'])
            text = paragraph['text'].lower()
            text = re.sub(r'-', '', text)
            text = re.sub(r'/', ' ', text)

            data.append((text, { 'entities': entities }))

    if max_docs > 0:
        data = data[:max_docs]

    nof_train = int(round(len(data)*train_split))
    train_data = data[:nof_train]
    test_data = data[nof_train:]
    labels = list(set(labels))
    
    return labels, train_data, test_data

def get_gold_data(train_split = 0.7, max_docs = 0):
    goldjson = json.loads(open(join('data', 'gold.json'), encoding='utf-8').read())
  
    data = []
    labels = []
    for doc in goldjson['documents']:
        for passage in doc['passages']:
            text = passage['text'].lower()
            text = re.sub(r'-', '', text)
            text = re.sub(r'/', ' ', text)
            entities = []
            for entity in passage['annotations']:
                start = entity['locations'][0]['offset']
                end = start + entity['locations'][0]['length']
                entities.append((start, end, entity['infons']['type']))
                labels.append(entity['infons']['type'])

            data.append((text, { 'entities': entities }))

    if max_docs > 0:
        data = data[:max_docs]

    nof_train = int(round(len(data)*train_split))
    train_data = data[:nof_train]
    test_data = data[nof_train:]
    labels = list(set(labels))

    return labels, train_data, test_data

if __name__ == "__main__":
    get_silver_data()
