import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import openslide
from skimage import draw

from src.constants import NAME_CLASS_MAPPING

ANNOTATIONS_FOLDERS = ["./data/train/annotations", "./data/test/annotations"]
DESTITATION_FOLDER = ["./data/train/masks", "./data/test/masks"]


"""
masks/
  (patient-id)/
      0/        -- folder related to individual .svs file
        in.png     -- input image converted to .png from .svs
        out.png    -- combination of all the masks, has values ranging from 0 to 4, where 0 is for background and 1 to 4 is class
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


def main():
    for i, ann_folder in enumerate(ANNOTATIONS_FOLDERS):
        dest_folder = DESTITATION_FOLDER[i]
        print(f"processing {ann_folder} folder...")

        patients = sorted(os.listdir(ann_folder))
        for patient in patients:
            patient_ann_dir = os.path.join(ann_folder, patient)
            patient_dest_dir = os.path.join(dest_folder, patient)
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
                svs_img = openslide.OpenSlide(svs_file)
                img = svs_img.read_region((0, 0), 0, svs_img.level_dimensions[0])
                w, h = img.size

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

                    if name not in NAME_CLASS_MAPPING.keys():
                        if not name == "Ambiguous":
                            print(
                                f"got invalid class name ({name}) while processing {j + 1} ann. tag in {xml_file} xml file"
                            )

                        continue

                    class_idx = NAME_CLASS_MAPPING[name]
                    binary_mask = np.zeros((h, w))

                    regions_tag = ann.find("Regions")
                    if regions_tag is None:
                        continue

                    vertex_row_coords = []
                    vertex_col_coords = []

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

                        fill_row_coords, fill_col_coords = draw.polygon(
                            vertex_col_coords, vertex_row_coords, binary_mask.shape
                        )
                        binary_mask[fill_row_coords, fill_col_coords] = 255

                    cv2.imwrite(
                        os.path.join(masks_dest_dir, f"{class_idx}.png"),
                        binary_mask,
                    )

                for k in list(NAME_CLASS_MAPPING.values()):
                    if not os.path.exists(os.path.join(masks_dest_dir, f"{k}.png")):
                        cv2.imwrite(
                            os.path.join(masks_dest_dir, f"{k}.png"), np.zeros((h, w))
                        )

                multiclass_mask = np.zeros((h, w))

                for k in list(NAME_CLASS_MAPPING.values()):
                    binary_mask = cv2.imread(
                        os.path.join(masks_dest_dir, f"{k}.png"), cv2.IMREAD_GRAYSCALE
                    )
                    if binary_mask is None:
                        continue

                    binary_mask = (binary_mask > 127).astype(np.uint8)
                    multiclass_mask[binary_mask == 1] = k + 1

                cv2.imwrite(
                    os.path.join(masks_dest_dir, "out.png"),
                    np.array(multiclass_mask),
                )
                cv2.imwrite(
                    os.path.join(masks_dest_dir, "in.png"),
                    cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                )


if __name__ == "__main__":
    main()
