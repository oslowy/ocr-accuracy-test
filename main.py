import json
import sys
import xml.dom.minidom as dom
import accuracy.accuracy as accuracy


def extract_image_info(image_element):
    tagged_rectangle_elements = image_element.getElementsByTagName("taggedRectangle")

    return {e.getElementsByTagName("tag")[0].firstChild.data:
            dict(e.attributes.items())
            for e in tagged_rectangle_elements}


def ground_truth_dictionary(gt_xml_file):
    document = dom.parse(gt_xml_file)
    image_elements = document.getElementsByTagName("image")

    return {e.getElementsByTagName("imageName")[0].firstChild.data[4:-4]:  # Remove img/ prefix and .jpg file extension
            extract_image_info(e)
            for e in image_elements}


def extract_output_word_infos(one_image_ocr, version):
    if version == 'aws':
        return [detection
                for detection in one_image_ocr['TextDetections']
                if detection['Type'] == 'WORD']
    else:  # Google format
        return one_image_ocr[1:]


def truth_word_correlation(truth, outputs, image_name, version):
    return [(accuracy.extract_output_word(output_word_info, version),
                accuracy.locate_output_word_in_truth(output_word_info, truth[image_name], version))
            for output_word_info in extract_output_word_infos(outputs[image_name], version)]


def main():
    # Args: ground_truth_filename, output_file_path, output_list_filename, version
    args = sys.argv[1:]
    gt_filename = args[0]
    out_path = args[1]
    out_list = args[2]
    version = args[3]

    # Read ground truth file
    with open(gt_filename) as gt_xml_file:
        truth = ground_truth_dictionary(gt_xml_file)

    # Read output files by reading filenames in meta-list file
    with open(out_list) as out_list_file:
        output_filenames = [filename.rstrip('\n') for filename in out_list_file]

        outputs_contents = {filename: json.load(open(f"{out_path}/{filename}.txt"))
                            for filename in output_filenames}

    # Extract data from output formats
    outputs_data = {image_name: extract_output_word_infos(outputs_contents[image_name], version)
                    for image_name in outputs_contents}

    # Run accuracy check
    correlations = {image_name: truth_word_correlation(truth, outputs_data, image_name, version)
                    for image_name in outputs_data}

    print(correlations)


if __name__ == "__main__":
    main()
