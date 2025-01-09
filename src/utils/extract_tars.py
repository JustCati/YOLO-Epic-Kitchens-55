import os
import tarfile
import argparse


def main(args):
    root_path = args.root_path

    for f in sorted(os.listdir(root_path)):
        f = os.path.join(root_path, f, "object_detection_images")
        for tar in sorted(os.listdir(f)):
            tar_path = os.path.join(f, tar)
            out_path = os.path.splitext(tar_path)[0]

            if not tar.endswith(".tar"):
                continue

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with tarfile.open(tar_path) as tar:
                tar.extractall(path=out_path)
            os.remove(tar_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "EPIC-KITCHENS"))
    args = parser.parse_args()
    main(args)
