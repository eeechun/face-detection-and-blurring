import os
from PIL import Image


def pgm2png():
    att_img_flat = "att_img_flat"

    count = 0
    for file in os.listdir(att_img_flat):
        filename, extension = os.path.splitext(file)
        if extension == ".pgm":
            count += 1
            print(file)
            new_file = f"{att_img_flat}/{filename}.jpg"
            with Image.open(f"{att_img_flat}/{file}") as img:
                img.save(new_file)
    print(f"proceed {count} images")


if __name__ == "__main__":
    pgm2png()
