import json
from os import listdir
from os.path import isfile, join

ori_path = join('data', 'comm_use_subset')
min_path = join('data', 'comm_use_subset_small')

def read_article(path):
    with open(path) as f:
        d = json.load(f)
        paper_id = d['paper_id']
        title_data = d['metadata']['title']
        abstract_data = [a['text'] for a in d['abstract']]
        body_text_data = [t['text'] for t in d['body_text']] #a list of all text sections in article

    return { 'paper_id': paper_id, 'title': title_data, 'abstract': abstract_data, 'body': body_text_data }

def main():
    files = [f for f in listdir(ori_path) if isfile(join(ori_path, f))]

    for i, fname in enumerate(files):
        print('Document {} of {}'.format(i + 1, len(files)))

        file_path = join(ori_path, fname)
        texts = read_article(file_path)
        with open(join(min_path, fname), 'w') as fp:
            json.dump(texts, fp)

if __name__ == '__main__':
    main()
