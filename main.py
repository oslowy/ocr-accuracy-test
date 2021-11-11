import sys
import xml.dom.minidom as dom


def extract_image_info(image_element):
    tagged_rectangle_elements = image_element.getElementsByTagName("taggedRectangle")

    return [dict([('word', e.getElementsByTagName("tag")[0].firstChild.data)]
                 + e.attributes.items())
            for e in tagged_rectangle_elements]


def ground_truth_dictionary(gt_xml_file):
    document = dom.parse(gt_xml_file)
    image_elements = document.getElementsByTagName("image")

    return [dict([('image', e.getElementsByTagName("imageName")[0].firstChild.data),
                  ('words', extract_image_info(e))])
            for e in image_elements]


def main():
    args = sys.argv[1:]

    with open(args[0]) as gt_xml_file:
        truth = ground_truth_dictionary(gt_xml_file)

    print(truth)


if __name__ == "__main__":
    main()
