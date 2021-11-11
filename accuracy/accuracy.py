from shapely.geometry import Polygon

from accuracy.area_check import area_within_ratio


def locate_output_word_in_truth(output_word, truth_words, version):
    """
    :param version: Google or AWS output format
    :param output_word: From cloud OCR output.
    :param truth_words: From ground truth data.
    :return: The word that the output word correlates to for edit distance checking
    """
    overlap_threshold = 0.75

    for truth_word in truth_words:
        if area_within_ratio(extract_bounding_poly(output_word, version),
                             convert_bounding_poly(truth_word)) >= overlap_threshold:
            return truth_word['word']
    return False


def convert_bounding_poly(truth_word_info):
    x0, y0 = truth_word_info['x'], truth_word_info['y']
    width, height = truth_word_info['width'], truth_word_info['height']

    # Points returned in order going around the outside of the rectangle
    return Polygon([(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)])


def extract_bounding_poly(output_word_info, version):
    if version == 'aws':
        return Polygon([(point['X'], point['Y']) for point in output_word_info['Geometry']['Polygon']])
    else:  # Google format
        return Polygon([(point['x'], point['y']) for point in output_word_info['boundingPoly']['vertices']])


def extract_word(output_word_info, version):
    if version == 'aws':
        return output_word_info['DetectedText']
    else:  # Google format
        return output_word_info['description']
