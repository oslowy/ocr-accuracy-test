from shapely.geometry import Polygon

from accuracy.area_check import area_within_ratio


def locate_output_word_in_truth(output_word_info, truth_word_infos, version):
    """
    :param version: Google or AWS output format
    :param output_word_info: From cloud OCR output.
    :param truth_word_infos: From ground truth data.
    :return: The word that the output word correlates to for edit distance checking
    """
    within_truth_threshold = 0.75

    for truth_word_info in truth_word_infos:
        word = truth_word_info['word']
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
        return output_word_info['DetectedText']
    else:  # Google format
        return output_word_info['description']
