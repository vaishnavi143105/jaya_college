import os
import glob
import io
import random
import xml.etree.ElementTree as ET

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "ai", "dataset", "raw")
IMAGES_DIR = os.path.join(BASE_DIR, "ai", "dataset", "images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "ai", "dataset", "annotations")

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


# ============================================================
# CLASS MAP
# ============================================================

CLASS_MAP = {
    "yes": 1,
    "i": 2,
    "iloveyou": 3,
    "thankyou": 4,
    "need": 5,
    "help": 6,
    "no": 7,
    "sorry": 8,
    "hello": 9,
    "goodbye": 10,
    "please": 11,
}


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(filename):
    """
    Search recursively inside dataset/images
    for the image referenced by the XML.
    """

    matches = glob.glob(
        os.path.join(IMAGES_DIR, "**", filename),
        recursive=True
    )

    if matches:
        return matches[0]

    # Try different extensions
    base = os.path.splitext(filename)[0]

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        matches = glob.glob(
            os.path.join(IMAGES_DIR, "**", base + ext),
            recursive=True
        )

        if matches:
            return matches[0]

    return None


# ============================================================
# CREATE TF EXAMPLE
# ============================================================

def create_tf_example(xml_path):

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Could not read XML: {xml_path}")
        print(e)
        return None

    filename_node = root.find("filename")

    if filename_node is None or not filename_node.text:
        print(f"[WARN] No filename in {xml_path}")
        return None

    filename = filename_node.text.strip()

    # Find corresponding image
    image_path = find_image(filename)

    if image_path is None:
        print(f"[WARN] Image not found: {filename}")
        return None

    try:
        with tf.io.gfile.GFile(image_path, "rb") as fid:
            encoded_image_data = fid.read()

        image = Image.open(io.BytesIO(encoded_image_data))
        width, height = image.size

    except Exception as e:
        print(f"[ERROR] Could not open image: {image_path}")
        print(e)
        return None

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for member in root.findall("object"):

        name_node = member.find("name")

        if name_node is None:
            continue

        class_name = name_node.text.strip().lower()

        if class_name not in CLASS_MAP:
            print(
                f"[WARN] Unknown class '{class_name}' "
                f"in {xml_path}"
            )
            continue

        bndbox = member.find("bndbox")

        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text) / width
        xmax = float(bndbox.find("xmax").text) / width
        ymin = float(bndbox.find("ymin").text) / height
        ymax = float(bndbox.find("ymax").text) / height

        # Clamp values
        xmin = max(0.0, min(xmin, 1.0))
        xmax = max(0.0, min(xmax, 1.0))
        ymin = max(0.0, min(ymin, 1.0))
        ymax = max(0.0, min(ymax, 1.0))

        if xmin >= xmax or ymin >= ymax:
            print(f"[WARN] Invalid bounding box in {xml_path}")
            continue

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        classes_text.append(class_name.encode("utf8"))
        classes.append(CLASS_MAP[class_name])

    if not classes:
        print(f"[WARN] No valid objects in {xml_path}")
        return None

    image_format = b"jpeg"

    if filename.lower().endswith(".png"):
        image_format = b"png"

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={

                "image/height":
                    dataset_util.int64_feature(height),

                "image/width":
                    dataset_util.int64_feature(width),

                "image/filename":
                    dataset_util.bytes_feature(
                        filename.encode("utf8")
                    ),

                "image/source_id":
                    dataset_util.bytes_feature(
                        filename.encode("utf8")
                    ),

                "image/encoded":
                    dataset_util.bytes_feature(
                        encoded_image_data
                    ),

                "image/format":
                    dataset_util.bytes_feature(
                        image_format
                    ),

                "image/object/bbox/xmin":
                    dataset_util.float_list_feature(xmins),

                "image/object/bbox/xmax":
                    dataset_util.float_list_feature(xmaxs),

                "image/object/bbox/ymin":
                    dataset_util.float_list_feature(ymins),

                "image/object/bbox/ymax":
                    dataset_util.float_list_feature(ymaxs),

                "image/object/class/text":
                    dataset_util.bytes_list_feature(
                        classes_text
                    ),

                "image/object/class/label":
                    dataset_util.int64_list_feature(
                        classes
                    ),
            }
        )
    )

    return tf_example


# ============================================================
# GENERATE RECORD
# ============================================================

def generate_record(xml_files, output_path):

    writer = tf.io.TFRecordWriter(output_path)

    count = 0

    for xml_file in xml_files:

        tf_example = create_tf_example(xml_file)

        if tf_example is not None:

            writer.write(
                tf_example.SerializeToString()
            )

            count += 1

            print(
                f"[OK] {count}/{len(xml_files)} "
                f"{os.path.basename(xml_file)}"
            )

    writer.close()

    print()
    print(
        f"[SUCCESS] Created {output_path} "
        f"with {count} examples."
    )

    return count


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MissVoice TFRecord Generator")
    print("=" * 60)

    # Find ALL XML files recursively
    xml_files = glob.glob(
        os.path.join(RAW_DIR, "**", "*.xml"),
        recursive=True
    )

    print(f"[INFO] XML files found: {len(xml_files)}")

    if len(xml_files) == 0:
        print("[ERROR] No XML files found!")
        exit(1)

    # Shuffle for train/test split
    random.seed(42)
    random.shuffle(xml_files)

    # 80% training / 20% testing
    split_index = int(len(xml_files) * 0.8)

    train_xml = xml_files[:split_index]
    test_xml = xml_files[split_index:]

    print(f"[INFO] Training XML files: {len(train_xml)}")
    print(f"[INFO] Testing XML files:  {len(test_xml)}")
    print()

    train_record = os.path.join(
        ANNOTATIONS_DIR,
        "train.record"
    )

    test_record = os.path.join(
        ANNOTATIONS_DIR,
        "test.record"
    )

    print("[INFO] Generating train.record...")
    train_count = generate_record(
        train_xml,
        train_record
    )

    print()
    print("[INFO] Generating test.record...")
    test_count = generate_record(
        test_xml,
        test_record
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Train examples: {train_count}")
    print(f"Test examples:  {test_count}")
    print(
        f"Total examples: {train_count + test_count}"
    )