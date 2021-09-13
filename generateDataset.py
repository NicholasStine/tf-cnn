from PIL import Image, ImageDraw
from os import listdir
import pandas as pd
import random

frames = listdir('frames')
templates = listdir('templates')

box_list = []

for n in range(1000):
    print('Generating Image: ', n)
    
    background = Image.open("frames/frame%d.jpg" % (n % len(frames)))
    template = Image.open("templates/template-%d.jpg" % (n % len(templates)))

    # Randomly resize the template
    resize_min = 0.1
    resize_max = 0.5
    template_resize_factor = resize_min + (resize_max - resize_min) * random.random()
    resized_template = template.resize((int(template.size[0] * template_resize_factor), int(template.size[1] * template_resize_factor)))

    # Randomly rotate the template
    rotated_template = resized_template.rotate(1 + (364 - 1) * random.random())

    # Randomly place the template on the frame
    x_min = 10 
    y_min = 10
    x_max = background.size[0] - resized_template.size[0] - 10
    y_max = background.size[1] - resized_template.size[1] - 10    
    template_x = int(x_min + (x_max - x_min) * random.random())
    template_y = int(y_min + (x_max - y_min) * random.random())

    background.paste(rotated_template, (template_x, template_y))

    background.save("combined/img%d.jpg" % n)
    box_list.append([template_x, template_y, template_x + resized_template.size[0], template_y + resized_template.size[1]])

print("Bounding Box Data: ", box_list)
box_dataframe = pd.DataFrame(box_list)
box_dataframe.to_csv("bounding_boxes.csv")

# Lets draw a few rectangles to verify the csv to image data
combined_list = listdir('combined')
opened_csv = pd.read_csv('bounding_boxes.csv')


# for n in range(2):
#     print('csv data: ', opened_csv.at[n, '0'])

#     combined_src = Image.open("combined/%s" % combined_list[n])
    
#     combined = ImageDraw.Draw(combined_src)
#     combined.rectangle([(opened_csv.at[n, '0'], opened_csv.at[n, '1']), (opened_csv.at[n, '2'], opened_csv.at[n, '3'])], width=10)
    
#     combined_src.show()