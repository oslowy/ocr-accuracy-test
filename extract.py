from xml.dom import minidom as dom


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
                for detection in one_image_ocr
                if detection['Type'] == 'WORD']
    else:  # Google format
        return one_image_ocr[1:]
