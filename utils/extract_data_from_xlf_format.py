import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to input rapid tilde .xlf file.",
        type=Path,
    )
    args = parser.parse_args()
    args.input = args.input.expanduser()
    return args


def main():
    args = get_args()
    tree = ET.parse(args.input)
    root = tree.getroot()
    m = re.match(r'\{(.*)\}', root.tag)
    namespaces = {"doc": m.group(1)}
    src_lang = root.get("srcLang")
    tgt_lang = root.get("trgLang")
    dir_ = args.input.parent
    src_path = dir_ / Path(src_lang)
    tgt_path = dir_ / Path(tgt_lang)
    with src_path.open('w') as src_f, tgt_path.open('w') as tgt_f:
        for segment in root.findall("./doc:file/doc:unit/doc:segment", namespaces):
            src = segment.find("./doc:source", namespaces).text
            tgt = segment.find("./doc:target", namespaces).text
            src_f.write(src + '\n')
            tgt_f.write(tgt + '\n')


if __name__ == "__main__":
    main()
