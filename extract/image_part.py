import os
from PIL import Image


def image_parts(image_path, output_dir, output_basename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(image_path)

    # print(img.filename)
    width, height = img.size
    x, y = int(width / 2) - 224 * 3, int(height / 2) - 224 * 3
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    parts = list()
    i = 0
    while i < 6:
        i = i + 1
        j = 0
        while j < 6:
            j = j + 1
            image = Image.new("L", (224, 224))
            _x = int(x + 224 * i)
            _y = int(y + 224 * j)
            image.paste(img.crop((_x, _y, _x + 224, _y + 224)), (0, 0, 224, 224))
            image_path = output_dir + "/" + output_basename + "_" + str(i) + "_" + str(j) + ".jpg"
            image.save(image_path)
            parts.append(image_path)

    for img in parts:
        m = Image.open(img)
        m = m.convert("RGB")
        m.save(img)

    return parts


if __name__ == '__main__':
    image_parts("/Users/philip.du/Documents/Projects/research/tea_cnn/samples/20190112/C-MPX0021-170950.jpg",
                "/Users/philip.du/Documents/Projects/research/tea_cnn/samples/20190112/parts",
                "C-MPX0021-170950.jpg")
