import json
import os
import sys
import xml.dom.minidom as dom
import accuracy.accuracy as accuracy
import csv


def extract_image_info(image_element):
    tagged_rectangle_elements = image_element.getElementsByTagName("taggedRectangle")

    # Key is sequence index, not target word, because target words are often duplicated
    return [dict([('word', e.getElementsByTagName("tag")[0].firstChild.data)] + e.attributes.items())
            for e in tagged_rectangle_elements]


def ground_truth_dictionary(gt_xml_file):
    document = dom.parse(gt_xml_file)
    image_elements = document.getElementsByTagName("image")

    # Key is filename because those are unique
    return {e.getElementsByTagName("imageName")[0].firstChild.data[4:-4]:  # Remove img/ prefix and .jpg file extension
                extract_image_info(e)
            for e in image_elements}


def extract_observed_word_infos(one_image_ocr, version):
    if version == 'aws':
        return [detection
                for detection in one_image_ocr['TextDetections']
                if detection['Type'] == 'WORD']
    else:  # Google format
        return one_image_ocr[1:]


def truth_word_correlation(truth, observations, image_name, version):
    return [(truth_word_info['word'],
             accuracy.locate_truth_word_in_observation(truth_word_info,
                                                       extract_observed_word_infos(observations[image_name],
                                                                                   version),
                                                       version))
            for truth_word_info in truth[image_name]]


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
    observations_contents = {filename[:-4]: json.load(open(f"{ocr_out_path}/{filename}"))
                             for filename in observation_filenames}

    # Extract data from observation formats
    observations_data = {image_name: extract_observed_word_infos(observations_contents[image_name], version)
                         for image_name in observations_contents}

    # Run accuracy check
    correlations = {image_name: truth_word_correlation(truth, observations_data, image_name, version)
                    for image_name in observations_data}
    match_scores = {image_name: [(*correlation, accuracy.accuracy_score(*correlation))
                                 for correlation in correlations[image_name]]
                    for image_name in correlations}

    # Output results
    scores_path = f"{ocr_out_path}_scores"
    if not os.path.exists(scores_path):
        os.makedirs(scores_path)

    for image in match_scores:
        with open(f"{scores_path}/{image}.csv", 'w') as scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(('truth', 'observed', 'score'))

            for score in match_scores[image]:
                writer.writerow(score)


if __name__ == "__main__":
    main()
