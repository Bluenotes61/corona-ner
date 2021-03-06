import json, re
from os import listdir
from os.path import isfile, join
import spacy
from spacy.pipeline import EntityRuler

collect_doc_paragraphs = True

def read_article_collect(path):
    with open(path) as f:
        d = json.load(f)
        paper_id = d['paper_id']
        text = d['title']
        for a in d['abstract']:
            if len(a) > 0:
                text = text + '\n' + a           
        for b in d['body']:
            if len(b) > 0:
                text = text + '\n' + b           

    if (len(text) > 1000000):
        text = text[:100000]

    return paper_id, [text]

def read_article(path):
    with open(path) as f:
        d = json.load(f)
        paper_id = d['paper_id']
        paragraphs = [d['title']] + d['abstract'] + d['body']

    return paper_id, paragraphs

def tag_article(nlp, article_path):
    if collect_doc_paragraphs:
      paper_id, paragraphs = read_article_collect(article_path)
    else:
      paper_id, paragraphs = read_article(article_path)

    denotated_paragraphs = []
    for paragraph in paragraphs:
        # Remove - and replace / with space
        paragraph = re.sub(r'-', '', paragraph)
        paragraph = re.sub(r'/', ' ', paragraph)

        doc = nlp(paragraph)

        if len(doc.ents) > 0:
            denotated_paragraphs.append(doc)

    return paper_id, denotated_paragraphs

def main():
    patterns = [
        { 'label': 'Disease_COVID-19', 'pattern': line[:-1] } 
        for line in open(join('data', 'covid19_list.txt'), encoding='utf8')
    ]
    patterns = patterns + [
        { 'label': 'Virus_SARS-CoV-2', 'pattern': line[:-1] } 
        for line in open(join('data', 'sars_list.txt'), encoding='utf8')
    ]

    nlp = spacy.blank('en')
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)

    files_path = join('data', 'comm_use_subset_small')
    files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    docs = []
    found = 0
    for i, filename in enumerate(files):
        paper_id, denotated_paragraphs = tag_article(nlp, join(files_path, filename))

        doc = { 'paper_id': paper_id, 'paragraphs': [] }
        if len(denotated_paragraphs) > 0:
            found = found + 1
            for paragraph in denotated_paragraphs:
                ents = []
                for ent in paragraph.ents:
                    ents.append({ 'label': ent.label_, 'text': ent.text, 'start': ent.start_char, 'end': ent.end_char })
                doc['paragraphs'].append({ 'text': paragraph.text, 'entities': ents })

            docs.append(doc)

        print('Document {} of {}. Denotated: {}'.format(i + 1, len(files), found), end = '\r')
    
    fname = 'corona_silver.json'
    if collect_doc_paragraphs:
        fname = 'corona_silver_collected.json'
    with open(join('data', fname), 'w') as fp:
        json.dump(docs, fp)
          
if __name__ == '__main__':
    main()
