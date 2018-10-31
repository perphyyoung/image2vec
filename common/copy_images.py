import shutil

root_image_path = './data/images/'
with open('./data/output/output.txt') as in_file:
    for line in in_file:
        image_name = line.split()
        image_path = root_image_path + image_name
        shutil.copyfile(image_path, 'images/' + image_name)
