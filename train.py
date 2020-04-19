import os, random, spacy, time, math
from helpers.kbhit import KBHit
from spacy.util import minibatch, compounding
from get_data import get_silver_data, get_gold_data
from matplotlib import pyplot as plt
from spacy.scorer import Scorer
from spacy.gold import GoldParse

n_iter = 100
evaluate_each = 5
train_split = 0.9
max_docs = 0 # 0 gets all available docs
dropout = 0.2
version = '1.2'
restart = True
data_type = 'gold'


def evaluate(nlp, TEST_DATA):
    scorer = Scorer()
    for text, annot in TEST_DATA:
        doc_gold_text = nlp.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot.get('entities'))
        pred_value = nlp(text)
        scorer.score(pred_value, gold)

    print('****************************')
    print('F-score: {}'.format(scorer.scores['ents_f']))
    print('P-score: {}'.format(scorer.scores['ents_p']))
    print('R-score: {}'.format(scorer.scores['ents_r']))
    print('Ents/type: {}'.format(scorer.scores['ents_per_type']))
    print('****************************\n')

    return scorer.scores

def main():
    # Enabling GPU somehow corrupts the training results
    #gpu = spacy.prefer_gpu()
    #print(gpu)
  
    if data_type == 'gold':
        LABELS, TRAIN_DATA, TEST_DATA = get_gold_data(train_split, max_docs)
    else:
        LABELS, TRAIN_DATA, TEST_DATA = get_silver_data(train_split, max_docs)

    model_dir = os.path.join('models', 'v' + version)
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    base_dir = model_dir
    if restart:
        base_dir = os.path.join('data', 'scispacy', 'en_core_sci_sm', 'en_core_sci_sm-0.2.4')

    nlp = spacy.load(base_dir)  # load existing spaCy model
    print("Loaded model '%s'" % base_dir)

    # create the built-in pipeline components and add them to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for label in LABELS:
        ner.add_label(label)

    lossarr = []
    fscorearr = []

    print('\nStart training {} of {} documents. Press q to stop training.\n'.format(len(TRAIN_DATA), len(TRAIN_DATA) + len(TEST_DATA)))
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    dobreak = False
    with nlp.disable_pipes(*other_pipes):  # only train NER
        last_time = time.time()
        kb = KBHit()
        for itn in range(n_iter):
            if dobreak:
                print('Stopping training')
                break
            
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for bi, batch in enumerate(batches):
                if kb.kbhit() and kb.getch() == 'q':
                    print('Stopping training')
                    dobreak = True
                    break

                texts, entities = zip(*batch)

                if (bi % 10) == 0:
                    print('Batch {} - size {}'.format(bi, len(texts)), end='\r')
                nlp.update(
                    texts,  # batch of texts
                    entities,  # batch of annotations
                    drop = dropout,  # dropout - make it harder to memorise data
                    losses = losses,
                )
            if 'ner' in losses:
                lossarr.append(losses['ner'])

            if ((itn + 1) % evaluate_each == 0):
                scores = evaluate(nlp, TEST_DATA)
                fscorearr.append(scores['ents_f'])

            curr_time = time.time()
            seconds = curr_time - last_time
            iter_mins = math.floor(seconds/60)
            iter_sec = math.floor(seconds - iter_mins * 60)
            last_time = curr_time
            eta_sec = (n_iter - itn) * seconds
            eta_hours = math.floor(eta_sec/3600)
            eta_mins = math.floor((eta_sec - eta_hours*3600)/60)
            print("Iteration {} of {} - Loss: {} - Time: {}m {}s - ETA: {}h {}m".format(itn + 1, n_iter, losses['ner'], iter_mins, iter_sec, eta_hours, eta_mins))

    info_file = open(os.path.join(model_dir, 'info.txt'), 'w')
    info_file.write('No train docs = {}\n'.format(len(TRAIN_DATA)))
    info_file.write('iterations = {}\n'.format(itn))
    info_file.write('train_split = {}\n'.format(train_split))
    info_file.write('dropout = {}\n'.format(dropout))
    info_file.write('F-score = {}\n'.format(fscorearr[-1]))
    info_file.close()
    
    nlp.to_disk(model_dir)
    print('Saved model to', model_dir)

    plt.plot(lossarr, linestyle='dashed', marker='o')
    plt.ylabel('NER loss')
    plt.show()

    plt.plot(fscorearr, linestyle='dashed', marker='o')
    plt.ylabel('F score')
    plt.show()

if __name__ == '__main__':
    main()
