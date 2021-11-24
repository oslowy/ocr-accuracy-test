from shapely.geometry import Polygon

from extract import extract_observed_word_infos


def area_within_ratio(observed_poly, target_poly):
    """
    In order to gracefully deal with segmentation errors, this function reports a high correlation
    score for an observed word even if its boundaries stretch over other target words, provided that
    a high proportion of the target word is inside the observed word's boundaries.

    :param observed_poly: The bounding polygon of an observed word, from cloud OCR output
    :param target_poly: The bounding polygon of the target word, from ground truth data
    that we are trying to find a match for
    :return: the best ratio of overlap area to total area of either the observed or truth word.
    """
    return max((observed_poly.intersection(target_poly)).area / observed_poly.area,
               (observed_poly.intersection(target_poly)).area / target_poly.area)


def locate_truth_word_in_observation(truth_word_info, observed_word_infos, version):
    """
    :param truth_word_info: From ground truth data.
    :param observed_word_infos: From cloud OCR output.
    :param version: Google or AWS observation data format
    :return: The word that the observed word correlates to for edit distance checking
    """
    within_truth_threshold = 0.75

    for observed_word_info in observed_word_infos:
        word = extract_observed_word(observed_word_info, version)
        within_truth_area = area_within_ratio(extract_observed_bounding_poly(observed_word_info, version),
                                              convert_truth_bounding_poly(truth_word_info))
        if within_truth_area >= within_truth_threshold:
            return word
    return False


def convert_truth_bounding_poly(word_info):
    x0, y0 = float(word_info['x']), float(word_info['y'])
    width, height = float(word_info['width']), float(word_info['height'])

    # Points returned in order going around the outside of the rectangle
    return Polygon([(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)])


def extract_observed_bounding_poly(word_info, version):
    if version == 'aws':
        return Polygon([(point['X'], point['Y']) for point in word_info['Geometry']['Polygon']])
    else:  # Google format
        return Polygon([(point['x'], point['y']) for point in word_info['boundingPoly']['vertices']])


def extract_observed_word(observed_word_info, version):
    if version == 'aws':
        word = observed_word_info['DetectedText']
    else:  # Google format
        word = observed_word_info['description']

    return word.encode('ascii', 'ignore').decode('ascii')


def truth_word_correlation(truth, observations, image_name, version):
    return [(truth_word_info['word'],
             locate_truth_word_in_observation(truth_word_info,
                                              extract_observed_word_infos(observations[image_name],
                                                                          version),
                                              version))
            for truth_word_info in truth[image_name]]
