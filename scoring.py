import csv
import os

from fuzzywuzzy.fuzz import ratio


def accuracy_score(truth_word, observed_word):
    return ratio(observed_word.lower()[:len(truth_word)],
                 truth_word.lower()) \
        if observed_word else 0


def get_scores(correlations):
    return {image_name: [(*correlation, accuracy_score(*correlation))
                         for correlation in correlations[image_name]]
            for image_name in correlations}


def get_averages(scores):
    return {image_name: ((sum(score_line[-1] for score_line in scores[image_name])
                          / len(scores[image_name]))
                         if scores[image_name] else 0)
            for image_name in scores}


def store_scores(scores, scores_path):
    for image in scores:
        with open(f"{scores_path}/{image}.csv", 'w') as scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(('truth', 'observed', 'score'))

            for score in scores[image]:
                writer.writerow(score)


def store_averages(averages, scores_path):
    with open(f"{scores_path}/AVERAGES.csv", 'w') as averages_file:
        writer = csv.writer(averages_file)
        writer.writerow(('image', 'average_score'))

        for image_name in averages:
            writer.writerow((image_name, averages[image_name]))


def make_scores_path(ocr_out_path):
    scores_path = f"{ocr_out_path}_scores"
    if not os.path.exists(scores_path):
        os.makedirs(scores_path)

    return scores_path
