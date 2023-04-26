from pathlib import Path
import shutil


def mvimg():

    att_img = Path("att_img")
    att_img_flat = Path("att_img_flat")
    if not att_img_flat.is_dir():
        att_img_flat.mkdir()
    assert att_img.is_dir()

    count = 0
    for p_name in att_img.iterdir():
        if p_name.is_dir():
            for img in p_name.iterdir():
                new_path = att_img_flat / Path(f"{p_name.name}_{img.name}")
                print(new_path)
                shutil.copy2(img, new_path)
                count += 1
        else:
            print("p_name is not dir")

    print(f"proceed {count} images")


if __name__ == "__main__":
    mvimg()
