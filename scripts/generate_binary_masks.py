import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import openslide
import skimage

from src.constants import NAME_CLASS_IDX_MAPPING

ANNOTATIONS_FOLDER = "./data/annotations"
DESTITATION_FOLDER = "./data/masks"


def main():
    patients = sorted(os.listdir(ANNOTATIONS_FOLDER))

    """
    masks/
      (patient-id)/
        0/        -- folder related to individual .svs file   
          0.png      -- binary mask for epithelial (class index 0)
          1.png      -- binary mask for lymphocyte (class index 1)
          2.png      -- binary mask for neutrophil (class index 2)
          3.png      -- binary mask for macrophage (class index 3)
        1/
          ...
        2/
          ...
        3/
          ...
    """

    for patient in patients:
        patient_ann_dir = os.path.join(ANNOTATIONS_FOLDER, patient)
        patient_dest_dir = os.path.join(DESTITATION_FOLDER, patient)
        os.makedirs(patient_dest_dir, exist_ok=True)

        xml_files = sorted(
            [
                os.path.join(patient_ann_dir, f)
                for f in os.listdir(patient_ann_dir)
                if f.endswith(".xml")
            ]
        )

        for i, xml_file in enumerate(xml_files):
            masks_dest_dir = os.path.join(patient_dest_dir, str(i))
            os.makedirs(masks_dest_dir, exist_ok=True)

            svs_file = xml_file.replace(".xml", ".svs")
            img = openslide.OpenSlide(svs_file)

            tree = ET.parse(xml_file)
            root = tree.getroot()

            for j, ann in enumerate(root):
                attributes_tag = ann.find("Attributes")
                if attributes_tag is None:
                    continue

                attribute_tag = attributes_tag.find("Attribute")
                if attribute_tag is None:
                    continue

                name = attribute_tag.get("Name")
                if name is None:
                    continue

                if name not in NAME_CLASS_IDX_MAPPING.keys():
                    print(
                        f"got invalid class name ({name}) while processing {j + 1} ann. tag in {xml_file} xml file"
                    )
                    continue

                class_idx = NAME_CLASS_IDX_MAPPING[name]
                binary_mask = np.transpose(
                    np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size))
                )

                regions_tag = ann.find("Regions")
                if regions_tag is None:
                    continue

                for region_tag in regions_tag:
                    vertices = region_tag.find("Vertices")
                    if vertices is None:
                        continue

                    coords = np.zeros((len(vertices), 2))

                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib["X"]
                        coords[i][1] = vertex.attrib["Y"]

                    vertex_row_coords = coords[:, 0]
                    vertex_col_coords = coords[:, 1]

                    fill_row_coords, fill_col_coords = skimage.draw.polygon(
                        vertex_row_coords, vertex_col_coords, binary_mask.shape
                    )
                    binary_mask[fill_row_coords, fill_col_coords] = 255

                cv2.imwrite(
                    os.path.join(masks_dest_dir, f"{class_idx}.png"),
                    binary_mask,
                )


if __name__ == "__main__":
    main()
