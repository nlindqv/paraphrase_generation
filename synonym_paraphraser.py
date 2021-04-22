import pandas as pd
import numpy as np
import pickle5 as pickle
import stanza
import requests
from bs4 import BeautifulSoup

from utils.batch_loader import clean_str

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
from nltk.lm import KneserNeyInterpolated

from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options

POS = {'ADJ': 'av', 'ADV': 'ab', 'VERB': 'vb', 'NOUN' : 'nn'}

class SynonymParaphraser:
    def __init__(self, model=None, ngram=3):
        if True:
            stanza.download('sv') # download Swedish model
            self.nlp = stanza.Pipeline('sv') # initialize Swedish neural pipeline
            self.base_url = 'https://www.synonymer.se/sv-syn/'

        # Build Language Model from corpus
        if model is None:
            with open('kneyser_lm.pkl', 'rb') as f:
                self.model = pickle.load(f)

        else:
            self.model = KneserNeyInterpolated(ngram)
            sentences = np.loadtxt(corpus_file, dtype='U', delimiter='\n')
            text = [list(map(str.lower, word_tokenize(sent))) for sent in sentences]
            train_data, padded_sents = padded_everygram_pipeline(ngram, text)
            self.model.fit(train_data, padded_sents)

    def generate_paraphrases(self, source_file):
        # Read data and make a copy to store edited paraphrases
        source_data = pd.read_csv(source_file)['question1']
        paraphrases = source_data.copy()

        for i in range(1688, len(source_data)):
            # Clean source sentences and generate dependency parse treee
            source_data[i] = clean_str(source_data[i])
            doc = self.nlp(source_data[i])
            print(doc)

            # Iterate all words to find potential words to replace with synonyms
            candidate_words = []
            for word in doc.sentences[0].words:
                if word.upos in ["ADJ", "ADV", "NOUN", "VERB"] and word.feats:
                    candidate_word = {'word' : word.text,
                                      'index' : word.id-1,
                                      'POS' : word.upos}
                    valid_candidate = True
                    features = [feature.split('=') for feature in word.feats.split('|')]
                    for feature in features:
                        if feature[0] == 'VerbForm' and feature[1] == 'Part':
                            valid_candidate = False
                            break
                        candidate_word[feature[0]] = feature[1]
                    if valid_candidate:
                        candidate_words.append(candidate_word)


            replacements = 0
            best_candidate = {'word': '', 'index': 0, 'diff' : -np.inf}
            for j, candidate in enumerate(candidate_words):
                candidate_synonyms = self.get_synonyms(candidate['word'])

                if candidate_synonyms == None:
                    continue
                original = (candidate['word'], self.get_score(candidate['word'], candidate['index'], source_data[i]))
                best_synonym = original
                synonym_count = 0
                for synonym in candidate_synonyms:
                    synonym = self.get_inflection(candidate, synonym)
                    if synonym is None:
                        continue
                    synonym_count += 1
                    # Calculate score for the synonym and compare to the current best
                    score = self.get_score(synonym, candidate['index'], source_data[i])
                    if score > best_synonym[1]:
                        best_synonym = (synonym, score)

                    diff = score - original[1]

                    if best_candidate['diff'] < diff:
                        best_candidate['word'] = synonym
                        best_candidate['index'] = candidate['index']
                        best_candidate['diff'] = diff
                        print(f'New best candidate: {synonym} with score {diff}')

                # Build paraphrase sentence
                if best_synonym[0] != candidate['word']:
                    new_sentence = ''
                    for (k, w) in enumerate(source_data[i].split()):
                        if k == candidate['index'] and best_synonym[0] != w:
                            new_sentence += best_synonym[0]
                            replacements += 1
                            print(f'Replaced word {w} with {best_synonym[0]}')
                        else:
                            new_sentence += w
                        if k < len(doc.sentences[0].words)-1:
                            new_sentence += ' '
                    source_data[i] = new_sentence

            # Assure at least one word is replaced with a synonym
            if replacements == 0 and best_candidate['word'] != '':
                print(best_candidate.items())
                new_sentence = ''
                for (k, w) in enumerate(source_data[i].split()):
                    if k == best_candidate['index']:
                        new_sentence += best_candidate['word']
                    else:
                        new_sentence += w
                    if k < len(doc.sentences[0].words)-1:
                        new_sentence += ' '
                source_data[i] = new_sentence

            print(f'{i} sentences done')
            print(source_data[i])
            print(paraphrases[i])
            print('\n')
            with open('synonym_samples_final.txt', 'a') as f:
                f.write(source_data[i] + '\n')

        return source_data


    def get_inflection(self, word, synonym):
        pos = POS[word['POS']]
        url = f"https://ws.spraakbanken.gu.se/ws/karp/v4/query?q=extended||and|pos|equals|{POS[word['POS']]}||and|wf|equals|{synonym}&resource=saldom"
        response = requests.get(url).json()['hits']

        if response['total'] == 0:
            return None

        msd = self.word_grammar(word)
        for i in range(len(response['hits'])):
            if response['hits'][i]['_source']['FormRepresentations'][0]['baseform'] in synonym:
                word_forms = response['hits'][i]['_source']['WordForms']

                for j in range(len(word_forms)):
                    if word_forms[j]['msd'] == msd:
                        if word['POS'] == 'NOUN' and 'Gender' in word.keys():
                            inherent = 'n' if word['Gender'] == 'Neut' else 'u'
                            if inherent != response['hits'][i]['_source']['FormRepresentations'][0]['inherent']:
                                return None
                        return word_forms[j]['writtenForm']


    def get_synonyms(self, word):
        synonyms = set()

        url = self.base_url + word
        html_doc = requests.get(url).text
        soup = BeautifulSoup(html_doc, 'html.parser')
        soup = soup.find("div", {"id":"dict-default"})
        if soup == None:
            return None
        else:
            soup = soup.find("div", {"body"}).ul
        for synset in soup.find_all('li'):
            for syns in synset.find_all('ol', class_=lambda x: not x):
                for synonym in syns.find_all('a'):
                    if len(synonym.text.split()) > 1:
                        continue
                    synonyms.add(synonym.text)
        return synonyms

    def get_score(self, word, j, source_sentence):
        scores = []
        sentence_len = len(source_sentence.split())
        if sentence_len >= 3:
            if j >= 2:
                scores.append(self.model.logscore(word, source_sentence.split()[(j-2):(j-1)]))
            if j < sentence_len-2:
                scores.append(self.model.logscore(source_sentence.split()[j+2], [word, source_sentence.split()[j+1]]))
            if j >= 1 and j < sentence_len-1:
                scores.append(self.model.logscore(source_sentence.split()[j-1], [source_sentence.split()[j+1], word]))
        else:
            if j == 0:
                scores.append(self.model.logscore(source_sentence.split()[1], [word]))
            else:
                scores.append(self.model.logscore(word, [source_sentence.split()[0]]))
        score = sum(scores) / len(scores)
        return score

    def word_grammar(self, word):
        grammar = None
        if word['POS'] == 'ADJ':
            if 'Degree' not in word:
                return None
            if word['Degree'] == 'Pos':
                grammar = 'pos'
            elif word['Degree'] == 'Cmp':
                grammar = 'komp'
                if 'Case' in word.keys() and word['Case'] == 'Nom':
                    grammar = grammar + ' nom'
                else:
                    grammar = grammar + ' gen'
                return grammar
            elif word['Degree'] == 'Sup':
                grammar = 'super'
                if 'Case' in word.keys() and word['Case'] == 'Nom':
                    grammar = grammar + ' nom'
                else:
                    grammar = grammar + ' gen'
                return grammar

            if 'Definite' not in word:
                return None
            if word['Definite'] == 'Ind':
                grammar = grammar + ' indef'
            elif word['Definite'] == 'Def':
                grammar = grammar + ' def'

            if 'Number' in word.keys():
                if word['Number'] == 'Sing':
                    grammar = grammar + ' sg'
                elif word['Number'] == 'Plur':
                    grammar = grammar + ' pl'

            if 'Gender' in word.keys() and word['Gender'] == 'Neut':
                grammar = grammar + ' n nom'
            else:
                grammar = grammar + ' u nom'

        elif word['POS'] == 'ADV':
            if 'Degree' not in word:
                return None
            else:
                if word['Degree'] == 'Pos':
                    grammar = 'pos'
                elif word['Degree'] == 'Cmp':
                    grammar = 'komp'
                elif word['Degree'] == 'Sup':
                    grammar = 'super'

        elif word['POS'] == 'VERB':
            if word['VerbForm'] == 'Inf':
                grammar = 'inf'
            elif word['VerbForm'] == 'Sup':
                grammar = 'sup'
            elif 'Tense' in word.keys() and word['Tense'] == 'Past':
                grammar = 'pret ind'
            elif word['Mood'] == 'Ind':
                grammar = 'pres ind'
            elif word['Mood'] == 'Imp':
                grammar = 'imper'
                return grammar

            if 'Voice' in word.keys() and word['Voice'] == 'Act':
                grammar = grammar + ' aktiv'
            else:
                grammar = grammar + ' s-form'

            # if
        elif word['POS'] == 'NOUN':
            if 'Number' not in word.keys():
                return None
            if word['Number'] == 'Sing':
                grammar = 'sg'
            elif word['Number'] == 'Plur':
                grammar = 'pl'


            if 'Definite' not in word.keys():
                return None
            elif word['Definite'] == 'Ind':
                grammar = grammar + ' indef'
            elif word['Definite'] == 'Def':
                grammar = grammar + ' def'

            if word['Case'] == 'Gen':
                grammar = grammar + ' gen'
            else:
                grammar = grammar + ' nom'

        return grammar

def main():
    corpus_file = 'svwiki2_xsmall'
    source_file = 'datasets/test.csv'
    paraphraser = SynonymParaphraser(corpus_file)
    paraphrases = paraphraser.generate_paraphrases(source_file)


if __name__=='__main__':
    main()
