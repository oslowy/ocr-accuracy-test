from shapely.geometry import Polygon
from fuzzywuzzy.fuzz import ratio


def area_within_ratio(observed_poly, target_poly):
    return (observed_poly.intersection(target_poly)).area / observed_poly.area


def locate_truth_word_in_output(truth_word_info, output_word_infos, version):
    """
    :param truth_word_info: From ground truth data.
    :param output_word_infos: From cloud OCR output.
    :param version: Google or AWS output format
    :return: The word that the output word correlates to for edit distance checking
    """
    within_truth_threshold = 0.75

    for output_word_info in output_word_infos:
        word = extract_output_word(output_word_info, version)
        within_truth_area = area_within_ratio(extract_output_bounding_poly(output_word_info, version),
                                              convert_truth_bounding_poly(truth_word_info))
        if within_truth_area >= within_truth_threshold:
            return word
    return False


def convert_truth_bounding_poly(word_info):
    x0, y0 = float(word_info['x']), float(word_info['y'])
    width, height = float(word_info['width']), float(word_info['height'])

    # Points returned in order going around the outside of the rectangle
    return Polygon([(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)])


def extract_output_bounding_poly(word_info, version):
    if version == 'aws':
        return Polygon([(point['X'], point['Y']) for point in word_info['Geometry']['Polygon']])
    else:  # Google format
        return Polygon([(point['x'], point['y']) for point in word_info['boundingPoly']['vertices']])


def extract_output_word(output_word_info, version):
    if version == 'aws':
        word = output_word_info['DetectedText']
    else:  # Google format
        word = output_word_info['description']

    return word.encode('ascii', 'ignore').decode('ascii')


def accuracy_score(truth_word, output_word):
    return ratio(output_word.lower()[:len(truth_word)],
                 truth_word.lower()) \
        if output_word else 0
