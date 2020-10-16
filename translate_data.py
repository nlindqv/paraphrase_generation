import os
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from google.cloud import translate_v2 as translate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-s', '--source', default='data/questions.csv', help='Filename of data')
    parser.add_argument('-d', '--destination', default='data/questions_sv.csv', help='Filename where preprocessed data are saved')
    parser.add_argument('-df', '--destination-filtered', default='data/questions_sv_filtered_all.csv', help='Filename where preprocessed data are saved')
    parser.add_argument('-t', '--translate', default=True, help='Translate dataset (default: True)')
    parser.add_argument('-l', '--language', default='sv', help='Target language (default: sv)')
    parser.add_argument('-f', '--filter-identical', default=True, help='Filter out paraphrases identical to original (default: True)')

    args = parser.parse_args()

    if not os.path.exists('data/questions.csv'):
        raise FileNotFoundError("Data file wasn't found")

    # Load data and filter duplicates
    questions = pd.read_csv(args.source, index_col = 0)
    question_duplicates = questions[questions['is_duplicate']==1]
    # print(question_duplicates.shape)



    # Load or create output file
    if not os.path.exists(args.destination):
        output_questions = pd.DataFrame(columns=list(question_duplicates.columns))
    else:
        output_questions = pd.read_csv(args.destination, index_col = 0)

    if args.translate:
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\Niklas Lindqvist\Documents\master_project\GCP_KEY.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"/mnt/c/Users/Niklas/Documents/master/master_project/GCP_KEY_2.json"
        translate_client = translate.Client()
        chars = 0
        words = 0
        for i in tqdm(range(question_duplicates.shape[0])):
            # Read in new sentences
            q1 = question_duplicates.iloc[i]['question1']
            q2 = question_duplicates.iloc[i]['question2']

            qid1 = question_duplicates.iloc[i]['qid1']
            qid2 = question_duplicates.iloc[i]['qid2']


            if qid1 not in output_questions['qid1'].values or qid2 not in output_questions['qid2'].values:
                # Count tokens and characters translated
                chars += len(q1) + len(q2)
                words += len(q1.split()) + len(q2.split())

                # Translate sentences
                q1_out = translate_client.translate(q1, target_language=args.language)
                q2_out = translate_client.translate(q2, target_language=args.language)
                q1_sv = q1_out['translatedText']
                q2_sv = q2_out['translatedText']

                # Add translations
                output_questions = output_questions.append({'qid1': qid1, 'qid2': qid2, 'question1': q1_sv, 'question2': q2_sv, 'is_duplicate': 1}, ignore_index=True)
                if i % 5000 == 0:
                    print(f'Save {i} sentences to file')
                    output_questions.to_csv('data/questions_sv.csv')
        print(f'Number of translated words: {words}\nNumber of translated characters: {chars}\n')

    output_questions.to_csv(args.destination)

    if args.filter_identical:
        identical = []
        for i in tqdm(range(len(output_questions))):
            # Read in new sentences
            q1 = output_questions.iloc[i]['question1']
            q2 = output_questions.iloc[i]['question2']

            if q1 == q2:
                identical.append(i)
        print(f'There were {len(identical)} identical question pairs, which is {len(identical)*100/99900}.2f%.')
        output_questions = output_questions.drop(identical, axis=0)

    output_questions.to_csv(args.destination_filtered)
