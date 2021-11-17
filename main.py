import json
import os
import sys

from correlate import truth_word_correlation
from extract import ground_truth_dictionary, extract_observed_word_infos
from scoring import get_scores, get_averages, store_scores, store_averages, make_scores_path


def main():
    # Args: ground truth filename, observation file path, version ('google' or 'aws')
    args = sys.argv[1:]
    gt_filename = args[0]
    ocr_out_path = args[1]
    version = args[2]

    # Read ground truth file
    with open(gt_filename) as gt_xml_file:
        truth = ground_truth_dictionary(gt_xml_file)

    # Read all observation files in directory
    observation_filenames = os.listdir(ocr_out_path)
    observations_contents = {filename[-9:-4]: json.load(open(f"{ocr_out_path}/{filename}"))
                             for filename in observation_filenames}

    # Extract data from observation formats
    observations_data = {image_name: extract_observed_word_infos(observations_contents[image_name]['annotations'],
                                                                 version)
                         for image_name in observations_contents}

    # Run accuracy check
    correlations = {image_name: truth_word_correlation(truth, observations_data, image_name, version)
                    for image_name in observations_data}
    scores = get_scores(correlations)
    averages = get_averages(scores)

    # Output results
    scores_path = make_scores_path(ocr_out_path)
    store_scores(scores, scores_path)
    store_averages(averages, scores_path)

    # Print results
    print("Average accuracy scores by file:")
    for image_name in averages:
        print(f"{image_name}: {averages[image_name]}%")


if __name__ == "__main__":
    main()
