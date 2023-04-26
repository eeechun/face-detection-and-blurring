from pathlib import Path
import shutil
import random


def train_test_split():
    source_folder = Path("pixel/16")  # example
    # dst_folder = Path("blur/15")
    dst_folder = source_folder
    train_data = dst_folder / Path("train")
    test_data = dst_folder / Path("test")
    train_data.mkdir(exist_ok=True)
    test_data.mkdir(exist_ok=True)

    for p in range(1, 41):
        test_id = random.choices(range(1, 11), k=2)
        print(test_id)
        for i in range(1, 11):
            img_name = f"s{p}_{i}.png"
            if (source_folder / Path(img_name)).is_file():
                if i in test_id:
                    # copy to test folder
                    shutil.copy2(
                        (source_folder / Path(img_name)), test_data / Path(img_name)
                    )
                else:
                    # copy to train folder
                    shutil.copy2(
                        (source_folder / Path(img_name)), train_data / Path(img_name)
                    )
            else:
                print(f"error: file {img_name} not exist")


if __name__ == "__main__":
    train_test_split()