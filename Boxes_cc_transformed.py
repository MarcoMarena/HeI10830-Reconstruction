#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2
import glob


# In[ ]:


# Function to process image
def process_image(image_path, size, disk_diameter_ratio):
    # Open and resize the original image
    image = Image.open(image_path).resize(size)

    # Calculate the disk diameter and radius
    disk_diameter = size[0] * disk_diameter_ratio
    disk_radius = disk_diameter / 2

    # Create a mask for the circular area
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((size[0]/2 - disk_radius, size[1]/2 - disk_radius, size[0]/2 + disk_radius, size[1]/2 + disk_radius), fill=255)

    # Create a black background image
    solar_disk = Image.new('RGB', size, 'black')

    # Paste the original image onto the black background using the mask
    solar_disk.paste(image, mask=mask)

    return solar_disk, disk_radius

def detect_regions(image, grid_size, active_threshold, quiet_threshold, disk_radius, polar_dist, max_threshold, min_threshold):
    gray_image = image.convert("L")
    active_boxes, quiet_boxes, npch_boxes, pch_boxes = [], [], [], []
    center_x, center_y = image.width // 2, image.height // 2

    for i in range(0, image.width, grid_size[0]):
        for j in range(0, image.height, grid_size[1]):
            box = (i, j, i + grid_size[0], j + grid_size[1])
            cell = gray_image.crop(box)
            cell_array = np.array(cell)

            # Calculate the center of the box
            box_center_x = (box[0] + box[2]) // 2
            box_center_y = (box[1] + box[3]) // 2

            # Calculate distance from the image center
            distance = np.sqrt((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) + grid_size[0] * 0.5

            # Check if the box center is within the solar disk
            if distance < disk_radius:
                avg_intensity = np.mean(cell_array)
                max_intensity = np.max(cell_array)
                min_intensity = np.min(cell_array)
                if avg_intensity < quiet_threshold and min_intensity <= min_threshold:
                    if box_center_y <= polar_dist or box_center_y >= image.height - polar_dist:
                        pch_boxes.append(box)
                    else:
                        npch_boxes.append(box)
                elif avg_intensity > active_threshold and max_intensity >= max_threshold:
                    active_boxes.append(box)
                else:
                    quiet_boxes.append(box)

    return active_boxes, quiet_boxes, npch_boxes, pch_boxes


# Function to extract date from filename
def extract_date_from_filename(filename):
    match = re.search(r'\d{8}', filename)
    return match.group(0) if match else 'Unknown Date'

# Function to visualize and save images with title
# def visualize_and_save_images_with_title(ground_truth_image, generated_image, grid_size, difference_image, active_boxes, quiet_boxes, npch_boxes, pch_boxes, disk_radius, save_path, title, DISK_RADIUS_RATIO):
#     processed_ground_truth = ground_truth_image.copy()
#     draw_boxes(processed_ground_truth, active_boxes, "red", line_width=8)
#     draw_boxes(processed_ground_truth, npch_boxes, "blue", line_width=8)
#     draw_boxes(processed_ground_truth, pch_boxes, "yellow", line_width=8)
#     draw = ImageDraw.Draw(processed_ground_truth)
#     draw_grid_within_disk(draw, disk_radius, grid_size, processed_ground_truth.width, processed_ground_truth.height, 2, DISK_RADIUS_RATIO)

#     processed_generated = generated_image.copy()
#     draw_boxes(processed_generated, active_boxes, "red", line_width=8)
#     draw_boxes(processed_generated, npch_boxes, "blue", line_width=8)
#     draw_boxes(processed_generated, pch_boxes, "yellow", line_width=8)

#     trio_image = Image.new('RGB', (processed_ground_truth.width * 3, processed_ground_truth.height))
#     trio_image.paste(processed_ground_truth, (0, 0))
#     trio_image.paste(processed_generated, (processed_ground_truth.width, 0))
#     trio_image.paste(difference_image, (processed_ground_truth.width * 2, 0))
    
#     draw.text((10, 10), titles[0], (255, 0, 0), font=font)
#     draw.text((processed_ground_truth.width + 10, 10), titles[1], (255, 255, 255), font=font)
#     draw.text((processed_ground_truth.width * 2 + 10, 10), titles[2], (255, 255, 255), font=font)
    
#     trio_image.save(save_path)
    
#     plt.figure(figsize=(20, 10))
#     plt.imshow(trio_image)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

def visualize_and_save_images_with_title(ground_truth_image, generated_image, grid_size, difference_image, active_boxes, quiet_boxes, npch_boxes, pch_boxes, disk_radius, save_path, title, DISK_RADIUS_RATIO):
    # Titles for each image
    titles = ["Ground Truth", "Generated", "Difference Map"]

    # Process the ground truth image
    processed_ground_truth = ground_truth_image.copy()
    draw_boxes(processed_ground_truth, active_boxes, "red", line_width=8)
    draw_boxes(processed_ground_truth, npch_boxes, "blue", line_width=8)
    draw_boxes(processed_ground_truth, pch_boxes, "yellow", line_width=8)
    draw = ImageDraw.Draw(processed_ground_truth)
    draw_grid_within_disk(draw, disk_radius, grid_size, processed_ground_truth.width, processed_ground_truth.height, 2, DISK_RADIUS_RATIO)


    # Process the generated image
    processed_generated = generated_image.copy()
    draw_boxes(processed_generated, active_boxes, "red", line_width=8)
    draw_boxes(processed_generated, npch_boxes, "blue", line_width=8)
    draw_boxes(processed_generated, pch_boxes, "yellow", line_width=8)
    
    # Create a new image to hold the three images
    trio_image = Image.new('RGB', (processed_ground_truth.width * 3, processed_ground_truth.height))
    trio_image.paste(processed_ground_truth, (0, 0))
    trio_image.paste(processed_generated, (processed_ground_truth.width, 0))
    trio_image.paste(difference_image, (processed_ground_truth.width * 2, 0))

    # Create draw object for trio_image
    draw = ImageDraw.Draw(trio_image)
     
    # Define font for the title
    font = ImageFont.truetype("/project/bs644/mm243/GAN_Solar/Font/Roboto-Bold.ttf", 60)

    # Function to draw centered text
    def draw_centered_text(image, text, font, y, left_x, right_x):
        text_width, text_height = draw.textsize(text, font=font)
        x = left_x + (right_x - left_x - text_width) // 2
        draw.text((x, y), text, fill="white", font=font)

    # Draw titles
    image_width = processed_ground_truth.width
    for i, title in enumerate(titles):
        draw_centered_text(trio_image, title, font, 10, i * image_width, (i + 1) * image_width)

    # Save the final image
    trio_image.save(save_path)

    # Display the image using matplotlib
#     plt.figure(figsize=(20, 10))
#     plt.imshow(trio_image)
#     plt.axis('off')
#     plt.show()

    
# Function to draw grid within the sun disk
def draw_grid_within_disk(draw, disk_radius, grid_size, image_width, image_height, line_width, DISK_RADIUS_RATIO):
    for i in range(0, image_width, grid_size[0]):
        for j in range(0, image_height, grid_size[1]):
            center_of_box = (i + grid_size[0] // 2, j + grid_size[1] // 2)
            distance_to_center = np.sqrt((center_of_box[0] - image_width // 2) ** 2 + (center_of_box[1] - image_height // 2) ** 2)
            if distance_to_center < disk_radius * DISK_RADIUS_RATIO:
                draw.rectangle((i, j, i + grid_size[0], j + grid_size[1]), outline="green", width=line_width)
    
# Function to draw boxes on an image
def draw_boxes(image, boxes, color, line_width=5):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=color, width=line_width)
        


# In[ ]:


# Function to bin the image
def bin_image(image, bin_size):
    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Initialize the binned image
    binned_image = np.zeros((image_array.shape[0] // bin_size, image_array.shape[1] // bin_size))

    # Iterate over the image to create the binned version
    for i in range(0, image_array.shape[0], bin_size):
        for j in range(0, image_array.shape[1], bin_size):
            bin_mean = np.mean(image_array[i:i+bin_size, j:j+bin_size])
            binned_image[i // bin_size, j // bin_size] = bin_mean

    return binned_image

# Function to calculate RMSE on binned images
def binned_rmse(image1, image2):
    return np.sqrt(mean_squared_error(image1, image2))

# Function to calculate MAE on binned images
def binned_mae(image1, image2):
    return mean_absolute_error(image1, image2)

# Function to calculate CC on binned images
def binned_cc(image1, image2):
    return pearsonr(image1.flatten(), image2.flatten())[0]

def cc(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Subtract the mean from x and y
    x_m = x - mean_x
    y_m = y - mean_y

    # Calculate numerator and denominators
    num = np.sum(x_m * y_m)
    den_x = np.sum(x_m ** 2)
    den_y = np.sum(y_m ** 2)

    # Calculate and return the correlation coefficient
    return num / np.sqrt(den_x * den_y)

def cc_fd(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Subtract the mean from x and y
    x_m = x - mean_x
    y_m = y - mean_y

    # Calculate numerator and denominators
    num = np.sum(x_m * y_m)
    den_x = np.sum(x_m ** 2)
    den_y = np.sum(y_m ** 2)

    # Calculate and return the correlation coefficient
    return num / np.sqrt(den_x * den_y)

# Function to calculate PPE10 on binned images
def binned_ppe10(image1, image2):
    # Calculate the percentage error between the two images
    # Avoid division by zero by adding a small constant to the denominator
    epsilon = 1e-10
    percentage_errors = np.abs(image1 - image2) / (np.abs(image2) + epsilon) * 100

    # Calculate the percentage of pixels with an error less than 10%
    pixels_less_than_10 = np.sum(percentage_errors < 10)
    total_pixels = np.size(percentage_errors)

    return (pixels_less_than_10 / total_pixels) * 100

def binned_re(image1, image2):

    # Filter out NaN values
    mask = ~np.isnan(image1) & ~np.isnan(image2)

    # Calculate relative errors where mask is True
    relative_errors = np.abs(image1[mask] - image2[mask]) / (image1[mask] + 0.0001)

    # Return the mean of the relative errors
    return np.mean(relative_errors)

def re_region(image1, image2):
    num = np.sqrt(np.sum((image1 - image2) ** 2))
    den = np.sqrt(np.sum(image1 ** 2))
    return num / den

def re_vector(image1, image2):
    num = np.sqrt(np.sum((image1 - image2) ** 2))
    den = np.sqrt(np.sum(image1 ** 2))
    return num / den

# Main function to calculate metrics
def calculate_metrics(image1, image2):
    bin_size = 8
    binned_1 = bin_image(image1, bin_size)
    binned_2 = bin_image(image2, bin_size)

    rmse = binned_rmse(binned_1, binned_2)
    mae = binned_mae(binned_1, binned_2)
    cc = binned_cc(binned_1, binned_2)
    ppe10 = binned_ppe10(binned_1, binned_2)
    re = binned_re(binned_1, binned_2)

    return rmse, mae, cc, ppe10, re

# Function to compute metrics
def acc(image1, image2, date):

    image_metrics = {'cc': [], 're': [], 'ppe10': [], 'mae': [], 'rmse': [], 'date': date}
    
    image_metrics['cc'].append(binned_cc(image1, image2))
    #image_metrics['re'].append(binned_re(image1, image2))
    image_metrics['re'].append(re_region(image1, image2))
    image_metrics['ppe10'].append(binned_ppe10(image1, image2))
    image_metrics['mae'].append(binned_mae(image1, image2))
    image_metrics['rmse'].append(binned_rmse(image1, image2))

    return image_metrics

def calculate_average_metrics(metrics_dict):
    # Initialize a dictionary to store the aggregated metrics
    agg_metrics = {'cc': [], 're': [], 'ppe10': [], 'mae': [], 'rmse': []}

    # Variables to keep track of the best CC value and its corresponding date
    best_cc_value = float('-inf')  # Initialize to negative infinity
    best_cc_date = None

    # Aggregate all metrics across dates and find the best CC value
    for date, date_metrics in metrics_dict.items():
        for metric in date_metrics:
            for key in agg_metrics:
                value = metric[key] if isinstance(metric[key], list) else [metric[key]]
                agg_metrics[key].extend(value)

                # Update the best CC value and date if necessary
                if key == 'cc' and max(value) > best_cc_value:
                    best_cc_value = max(value)
                    best_cc_date = date

    # Calculate the average for each metric using NumPy
    avg_metrics = {key: np.mean(agg_metrics[key]) for key in agg_metrics}

    return avg_metrics, best_cc_date


def find_best_date(metrics_dict):
    best_date = ''
    best_score = float('inf')

    for date, metrics in metrics_dict.items():
        # Example scoring: sum of all metrics (lower is better)
        score = sum(metric['cc'] + metric['re'] + metric['ppe10'] + metric['mae'] + metric['rmse'] for metric in metrics)

        if score < best_score:
            best_score = score
            best_date = date

    return best_date

def plot_trio_image(date, directory, region_type):
    # Construct the file path for the trio image
    # This depends on your file naming convention
    filename = f"{date}_trio.png"
    file_path = os.path.join(directory, filename)

    # Load and plot the image
#     img = mpimg.imread(file_path)
#     plt.imshow(img)
#     plt.title(f"Trio Image for {region_type} on {date}")
#     plt.axis('off')  # Turn off axis
#     plt.show()
    
def reshape(arr):
    return np.resize(arr, (np.size(arr) * np.size(arr[0]), 1))

def region_metrics(boxes, grey_gt, grey_gen, BIN_SIZE, fl_gt, fl_gen, fl_bin_gt, fl_bin_gen, image_date, metrics_2, metrics_4):
    
    count = 0
    for box in tqdm(boxes):
        cropped_gt = grey_gt.crop(box)
        cropped_generated = grey_gen.crop(box)

        gt_array = np.array(cropped_gt)
        gen_array = np.array(cropped_generated)

        bin_gt = bin_image(cropped_gt, BIN_SIZE)
        bin_gen = bin_image(cropped_generated, BIN_SIZE)

        fl_gt.append(gt_array.flatten())
        fl_gen.append(gen_array.flatten())

        fl_bin_gt.append(bin_gt.flatten())
        fl_bin_gen.append(bin_gen.flatten())

        # binned - Table 2
        metrics_bin = acc(bin_gt, bin_gen, image_date)
        metrics_2[image_date].append(metrics_bin)

        # not binned - Table 4
        metrics = acc(cropped_gt, cropped_generated, image_date)
        metrics_4[image_date].append(metrics)

        count += 1
    return fl_gt, fl_gen, fl_bin_gt, fl_bin_gen, metrics_2, metrics_4, count


# In[ ]:


def accuracy_table(directory, save_directory, BIN_SIZE = 16, GRID_SIZE = (128, 128), 
                   ACTIVE_THRESHOLD = 110, QUIET_THRESHOLD = 65, MAX_THRESHOLD = 200,
                   MIN_THRESHOLD = 10, DISK_DIAMETER_RATIO = 0.9, DISK_RADIUS_RATIO = 0.9, 
                   POLAR_REGION_RATIO = 0.19, IMAGE_SIZE = (1024, 1024)):

    BINNING = (BIN_SIZE, BIN_SIZE)
    
    grid_s = GRID_SIZE[0]
    
    # Initialize metrics dictionaries for each region
    ar_metrics_2, qr_metrics_2, npch_metrics_2, pch_metrics_2 = {}, {}, {}, {}
    ar_metrics_4, qr_metrics_4, npch_metrics_4, pch_metrics_4 = {}, {}, {}, {}

    polar_value = POLAR_REGION_RATIO * IMAGE_SIZE[1]
    
    # Initialize metric dictionary for full disk
    fd_dict = {}
    fd_bin_dict = {}
    
    ar_fl_gt, qr_fl_gt, npch_fl_gt, pch_fl_gt = [], [], [], []
    ar_fl_gen, qr_fl_gen, npch_fl_gen, pch_fl_gen = [], [], [], []
    
    ar_fl_bin_gt, qr_fl_bin_gt, npch_fl_bin_gt, pch_fl_bin_gt = [], [], [], []
    ar_fl_bin_gen, qr_fl_bin_gen, npch_fl_bin_gen, pch_fl_bin_gen = [], [], [], []
    
    
    ar_count, qr_count, npch_count, pch_count = 0, 0, 0, 0
    
    for filename in tqdm(sorted(os.listdir(directory)), desc="Processing Images"):
        if filename.endswith("_real_B.png"):
            ground_truth_path = os.path.join(directory, filename)
            
            full_disk = {'cc': [], 're': [], 'ppe10': [], 'mae': [], 'rmse': [], 'date': []}
            full_disk_bin = {'cc': [], 're': [], 'ppe10': [], 'mae': [], 'rmse': [], 'date': []}

            ground_truth_image, disk_radius = process_image(ground_truth_path, IMAGE_SIZE, DISK_DIAMETER_RATIO)

            ar_boxes, qr_boxes, npch_boxes, pch_boxes = detect_regions(ground_truth_image, GRID_SIZE, ACTIVE_THRESHOLD, 
                                                                       QUIET_THRESHOLD, disk_radius, polar_value, MAX_THRESHOLD, MIN_THRESHOLD)

            generated_path = os.path.join(directory, filename.replace('_real_B', '_fake_B'))
            generated_image, _ = process_image(generated_path, IMAGE_SIZE, DISK_DIAMETER_RATIO)

            difference_image = ImageChops.difference(ground_truth_image, generated_image)

            # Visualize and save images with boxes
            image_date = extract_date_from_filename(filename)
            trio_save_path = os.path.join(save_directory, filename.replace('_real_B.png', '_trio.png'))

            visualize_and_save_images_with_title(ground_truth_image, generated_image, GRID_SIZE, difference_image, ar_boxes, qr_boxes, npch_boxes, pch_boxes, disk_radius, trio_save_path, image_date, DISK_RADIUS_RATIO)
            
            grey_gt = ground_truth_image.convert("L")
            grey_gen = generated_image.convert("L")
            
            grey_gt_array = np.array(grey_gt)
            grey_gen_array = np.array(grey_gen)
            
            grey_gt_array = grey_gt_array.astype(float)
            grey_gen_array = grey_gen_array.astype(float)

            grey_gt_array = np.clip(grey_gt_array, np.log(50),  np.log(2500))
            grey_gen_array = np.clip(grey_gen_array,  np.log(50),  np.log(2500))
            
            grey_gt_array = np.exp(grey_gt_array)
            grey_gen_array = np.exp(grey_gen_array)
            
            center_x, center_y = grey_gt_array.shape[0] / 2, grey_gt_array.shape[1] / 2

            # Create a grid of x, y coordinates
            xx, yy = np.meshgrid(np.arange(grey_gt_array.shape[0]), np.arange(grey_gt_array.shape[1]))

            # Calculate the squared distance from the center
            dist_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2

            # Create a mask for values outside the disk
            outside_disk_mask = dist_squared > disk_radius ** 2

            # Apply the mask
            grey_gt_array[outside_disk_mask] = np.NaN
            grey_gen_array[outside_disk_mask] = np.NaN
            
            # binned flatten - Table 1
            ar_metrics_2[image_date] = []
            qr_metrics_2[image_date] = []
            npch_metrics_2[image_date] = []
            pch_metrics_2[image_date] = []

            # flatten - Table 3
            ar_metrics_4[image_date] = []
            qr_metrics_4[image_date] = []
            npch_metrics_4[image_date] = []
            pch_metrics_4[image_date] = []
            
            # Full Disk
            fd_gt = grey_gt_array.flatten()
            fd_gen = grey_gen_array.flatten()
            
            fd_gt = fd_gt[~np.isnan(fd_gt)]
            fd_gen = fd_gen[~np.isnan(fd_gen)]
            
            fd_cc = cc_fd(fd_gt, fd_gen)
            #fd_re = np.nanmean(np.abs(fd_gt - fd_gen) / (fd_gt + 0.00001))
            fd_re = re_vector(fd_gt, fd_gen)
            fd_ppe10 = binned_ppe10(fd_gt, fd_gen)
            fd_mae = mean_absolute_error(fd_gt, fd_gen)
            fd_rmse = np.sqrt(mean_squared_error(fd_gt, fd_gen))
            
            full_disk.update({'cc': fd_cc, 're': fd_re, 'ppe10': fd_ppe10, 'mae': fd_mae, 'rmse': fd_rmse})
            fd_dict[image_date] = []
            fd_dict[image_date].append(full_disk)
            # Full Disk Binned
            
            fd_bin_gt = bin_image(grey_gt_array, BIN_SIZE)
            fd_bin_gen = bin_image(grey_gen_array, BIN_SIZE)
            
            fd_bin_gt = fd_bin_gt.flatten()
            fd_bin_gen = fd_bin_gen.flatten()
            
            fd_bin_gt = fd_bin_gt[~np.isnan(fd_bin_gt)]
            fd_bin_gen = fd_bin_gen[~np.isnan(fd_bin_gen)]
            
            fd_bin_cc = cc_fd(fd_bin_gt, fd_bin_gen)
            #fd_bin_re = np.nanmean(np.abs(fd_bin_gt - fd_bin_gen) / (fd_bin_gt + 0.00001))
            fd_bin_re = re_vector(fd_bin_gt, fd_bin_gen)
            fd_bin_ppe10 = binned_ppe10(fd_bin_gt, fd_bin_gen)
            fd_bin_mae = mean_absolute_error(fd_bin_gt, fd_bin_gen)
            fd_bin_rmse = np.sqrt(mean_squared_error(fd_bin_gt, fd_bin_gen))
            
            full_disk_bin.update({'cc': fd_bin_cc, 're': fd_bin_re, 'ppe10': fd_bin_ppe10, 'mae': fd_bin_mae, 'rmse': fd_bin_rmse})
            fd_bin_dict[image_date] = []
            fd_bin_dict[image_date].append(full_disk_bin)
            
            
            # Calculate metrics for ACTIVE REGIONS
            for box in ar_boxes:
                cropped_gt = grey_gt.crop(box)
                cropped_generated = grey_gen.crop(box)

                gt_array = np.array(cropped_gt)
                gen_array = np.array(cropped_generated)
                
                gt_array = np.clip(gt_array, np.log(50),  np.log(2500))
                gen_array = np.clip(gen_array,  np.log(50),  np.log(2500))

                gt_array = np.exp(gt_array)
                gen_array = np.exp(gen_array)

                bin_gt = bin_image(cropped_gt, BIN_SIZE)
                bin_gen = bin_image(cropped_generated, BIN_SIZE)
                
                bin_gt = np.clip(bin_gt, np.log(50),  np.log(2500))
                bin_gen = np.clip(bin_gen,  np.log(50),  np.log(2500))

                bin_gt = np.exp(bin_gt)
                bin_gen = np.exp(bin_gen)

                ar_fl_gt.append(gt_array.flatten())
                ar_fl_gen.append(gen_array.flatten())

                ar_fl_bin_gt.append(bin_gt.flatten())
                ar_fl_bin_gen.append(bin_gen.flatten())

                # binned - Table 2
                ar_metrics_bin = acc(bin_gt, bin_gen, image_date)
                ar_metrics_2[image_date].append(ar_metrics_bin)

                # not binned - Table 4
                ar_metrics = acc(gt_array, gen_array, image_date)
                ar_metrics_4[image_date].append(ar_metrics)

                ar_count += 1
#             # Calculate metrics for quiet regions
#             qr_fl_gt, qr_fl_gen, qr_fl_bin_gt, qr_fl_bin_gen, qr_metrics_2, qr_metrics_4, qr_count = region_metrics(qr_boxes, qr_grey_gt, qr_grey_gen, BIN_SIZE, qr_fl_gt, qr_fl_gen, qr_fl_bin_gt, qr_fl_bin_gen, image_date, qr_metrics_2, qr_metrics_4)

            for box in npch_boxes:
                cropped_gt = grey_gt.crop(box)
                cropped_generated = grey_gen.crop(box)

                gt_array = np.array(cropped_gt)
                gen_array = np.array(cropped_generated)

                gt_array = np.clip(gt_array, np.log(50),  np.log(2500))
                gen_array = np.clip(gen_array,  np.log(50),  np.log(2500))

                gt_array = np.exp(gt_array)
                gen_array = np.exp(gen_array)

                bin_gt = bin_image(cropped_gt, BIN_SIZE)
                bin_gen = bin_image(cropped_generated, BIN_SIZE)
                
                bin_gt = np.clip(bin_gt, np.log(50),  np.log(2500))
                bin_gen = np.clip(bin_gen,  np.log(50),  np.log(2500))

                bin_gt = np.exp(bin_gt)
                bin_gen = np.exp(bin_gen)

                npch_fl_gt.append(gt_array.flatten())
                npch_fl_gen.append(gen_array.flatten())

                npch_fl_bin_gt.append(bin_gt.flatten())
                npch_fl_bin_gen.append(bin_gen.flatten())

                # binned - Table 2
                npch_metrics_bin = acc(bin_gt, bin_gen, image_date)
                npch_metrics_2[image_date].append(npch_metrics_bin)

                # not binned - Table 4
                npch_metrics = acc(gt_array, gen_array, image_date)
                npch_metrics_4[image_date].append(npch_metrics)

                npch_count += 1

            for box in pch_boxes:
                cropped_gt = grey_gt.crop(box)
                cropped_generated = grey_gen.crop(box)

                gt_array = np.array(cropped_gt)
                gen_array = np.array(cropped_generated)

                gt_array = np.clip(gt_array, np.log(50),  np.log(2500))
                gen_array = np.clip(gen_array,  np.log(50),  np.log(2500))

                gt_array = np.exp(gt_array)
                gen_array = np.exp(gen_array)

                bin_gt = bin_image(cropped_gt, BIN_SIZE)
                bin_gen = bin_image(cropped_generated, BIN_SIZE)
                
                bin_gt = np.clip(bin_gt, np.log(50),  np.log(2500))
                bin_gen = np.clip(bin_gen,  np.log(50),  np.log(2500))

                bin_gt = np.exp(bin_gt)
                bin_gen = np.exp(bin_gen)

                pch_fl_gt.append(gt_array.flatten())
                pch_fl_gen.append(gen_array.flatten())

                pch_fl_bin_gt.append(bin_gt.flatten())
                pch_fl_bin_gen.append(bin_gen.flatten())

                # binned - Table 2
                pch_metrics_bin = acc(bin_gt, bin_gen, image_date)
                pch_metrics_2[image_date].append(pch_metrics_bin)

                # not binned - Table 4
                pch_metrics = acc(gt_array, gen_array, image_date)
                pch_metrics_4[image_date].append(pch_metrics)

                pch_count += 1

    ar_fl_bin_gt = reshape(ar_fl_bin_gt)
    ar_fl_bin_gen = reshape(ar_fl_bin_gen)
    npch_fl_bin_gt = reshape(npch_fl_bin_gt)
    npch_fl_bin_gen = reshape(npch_fl_bin_gen)
    pch_fl_bin_gt = reshape(pch_fl_bin_gt)
    pch_fl_bin_gen = reshape(pch_fl_bin_gen)
#     qr_fl_bin_gt = reshape(qr_fl_bin_gt)
#     qr_fl_bin_gen = reshape(qr_fl_bin_gen)

    avg_full_disk, fd_date = calculate_average_metrics(fd_dict)
    avg_full_disk_bin, fd_bin_date = calculate_average_metrics(fd_bin_dict)
    
    
#     ar_fl_gt = reshape(ar_fl_gt)
#     ar_fl_gen = reshape(ar_fl_gen)  
# #     qr_fl_gt = reshape(qr_fl_gt)
# #     qr_fl_gen = reshape(qr_fl_gen)
#     npch_fl_gt = reshape(npch_fl_gt)
#     npch_fl_gen = reshape(npch_fl_gen)
#     pch_fl_gt = reshape(pch_fl_gt)
#     pch_fl_gen = reshape(pch_fl_gen)
    
    avg_ar_metrics_2, ar_2_date = calculate_average_metrics(ar_metrics_2)
    avg_qr_metrics_2, qr_2_date = calculate_average_metrics(qr_metrics_2)
    avg_npch_metrics_2, npch_2_date = calculate_average_metrics(npch_metrics_2)
    avg_pch_metrics_2, pch_2_date = calculate_average_metrics(pch_metrics_2)
    
#     avg_ar_metrics_4 = calculate_average_metrics(ar_metrics_4)
#     avg_qr_metrics_4 = calculate_average_metrics(qr_metrics_4)
#     avg_npch_metrics_4 = calculate_average_metrics(npch_metrics_4)
#     avg_pch_metrics_4 = calculate_average_metrics(pch_metrics_4)

#     print('--------Average 4 DONE---------')
    
    
    
#     best_ar_date = find_best_date(ar_metrics)
#     best_qr_date = find_best_date(qr_metrics)
#     best_npch_date = find_best_date(npch_metrics)
#     best_pch_date = find_best_date(pch_metrics)
    
#     # Assuming save_directory is where your trio images are saved
#     plot_trio_image(best_ar_date, save_directory, "AR")
#     plot_trio_image(best_qr_date, save_directory, "QR")
#     plot_trio_image(best_npch_date, save_directory, "NPCH")
#     plot_trio_image(best_pch_date, save_directory, "PCH")

    # Create a pandas Dataframe for full disk
    data_0 = {
        'Metric': ['CC', 'RE', 'PPE10', 'MAE', 'RMSE'],
        'Full Disk': list(avg_full_disk.values()),
        'Full Disk Binned': list(avg_full_disk_bin.values())
    }
    
    print('\n----------Full Disk Table----------\n', pd.DataFrame(data_0).transpose())
    print('FD date:', fd_date)
    print('FD bin date:', fd_bin_date)
    # Create a pandas DataFrame and display the table
    data_1 = {
        'Metric': ['CC', 'RE', 'PPE10', 'MAE', 'RMSE'],
        f'Active Region ({ar_count})': [cc(ar_fl_bin_gt, ar_fl_bin_gen), re_vector(ar_fl_bin_gt, ar_fl_bin_gen), binned_ppe10(ar_fl_bin_gt, ar_fl_bin_gen), mean_absolute_error(ar_fl_bin_gt, ar_fl_bin_gen), np.sqrt(mean_squared_error(ar_fl_bin_gt, ar_fl_bin_gen))], 
#         f'Quiet Region ({qr_count})': [cc(qr_fl_bin_gt, qr_fl_bin_gen), re_vector(qr_fl_bin_gt, qr_fl_bin_gen), binned_ppe10(qr_fl_bin_gt, qr_fl_bin_gen), mean_absolute_error(qr_fl_bin_gt, qr_fl_bin_gen), np.sqrt(mean_squared_error(qr_fl_bin_gt, qr_fl_bin_gen))],
        f'Polar Coronal Holes ({pch_count})': [cc(npch_fl_bin_gt, npch_fl_bin_gen), re_vector(npch_fl_bin_gt, npch_fl_bin_gen), binned_ppe10(npch_fl_bin_gt, npch_fl_bin_gen), mean_absolute_error(npch_fl_bin_gt, npch_fl_bin_gen), np.sqrt(mean_squared_error(npch_fl_bin_gt, npch_fl_bin_gen))],
        f'Non-Polar Coronal Holes ({npch_count})': [cc(pch_fl_bin_gt, pch_fl_bin_gen), re_vector(pch_fl_bin_gt, pch_fl_bin_gen), binned_ppe10(pch_fl_bin_gt, pch_fl_bin_gen), mean_absolute_error(pch_fl_bin_gt, pch_fl_bin_gen), np.sqrt(mean_squared_error(pch_fl_bin_gt, pch_fl_bin_gen))]}
    
    print('\n----------Binned Vectorized Table----------\n', pd.DataFrame(data_1).transpose())
    
    data_2 = {
        'Metric': ['CC', 'RE', 'PPE10', 'MAE', 'RMSE'],
        f'Active Region ({ar_count})': list(avg_ar_metrics_2.values()),
#         f'Quiet Region ({qr_count})': list(avg_qr_metrics_2.values()),
        f'Polar Coronal Holes ({pch_count})': list(avg_pch_metrics_2.values()),
        f'Non-Polar Coronal Holes ({npch_count})': list(avg_npch_metrics_2.values())
    }
    
    print('\n----------Binned Regions Table----------\n', pd.DataFrame(data_2).transpose())
    print()
    print('AR date:', ar_2_date)
    print('QR date:', qr_2_date)
    print('PCH date:', pch_2_date)
    print('NPCH date:', npch_2_date)
    
#     data_3 = {
#         'Metric': ['CC', 'RE', 'PPE10', 'MAE', 'RMSE'],
#         f'Active Region ({ar_count})': [cc(ar_fl_gt, ar_fl_gen), binned_re(ar_fl_gt, ar_fl_gen), binned_ppe10(ar_fl_gt, ar_fl_gen), mean_absolute_error(ar_fl_gt, ar_fl_gen), np.sqrt(mean_squared_error(ar_fl_gt, ar_fl_gen))], 
# #         f'Quiet Region ({qr_count})': [cc(qr_fl_gt, qr_fl_gen), binned_re(qr_fl_gt, qr_fl_gen), binned_ppe10(qr_fl_gt, qr_fl_gen), mean_absolute_error(qr_fl_gt, qr_fl_gen), np.sqrt(mean_squared_error(qr_fl_gt, qr_fl_gen))],
#         f'Polar Coronal Holes ({pch_count})': [cc(npch_fl_gt, npch_fl_gen), binned_re(npch_fl_gt, npch_fl_gen), binned_ppe10(npch_fl_gt, npch_fl_gen), mean_absolute_error(npch_fl_gt, npch_fl_gen), np.sqrt(mean_squared_error(npch_fl_gt, npch_fl_gen))],
#         f'Non-Polar Coronal Holes ({npch_count})': [cc(pch_fl_gt, pch_fl_gen), binned_re(pch_fl_gt, pch_fl_gen), binned_ppe10(pch_fl_gt, pch_fl_gen), mean_absolute_error(pch_fl_gt, pch_fl_gen), np.sqrt(mean_squared_error(pch_fl_gt, pch_fl_gen))]}
#     print('\n----------Vectorized Table----------\n', pd.DataFrame(data_3).transpose())    
        
#     data_4 = {
#         'Metric': ['CC', 'RE', 'PPE10', 'MAE', 'RMSE'],
#         f'Active Region ({ar_count})': list(avg_ar_metrics_4.values()),
# #         f'Quiet Region ({qr_count})': list(avg_qr_metrics_4.values()),
#         f'Polar Coronal Holes ({pch_count})': list(avg_pch_metrics_4.values()),
#         f'Non-Polar Coronal Holes ({npch_count})': list(avg_npch_metrics_4.values())
#     }
#     print('\n----------Regions Table----------\n', pd.DataFrame(data_4).transpose())
    
    return pd.DataFrame(data_0).transpose(), pd.DataFrame(data_1).transpose(), pd.DataFrame(data_2).transpose()#, pd.DataFrame(data_3).transpose(), pd.DataFrame(data_4).transpose()
#     # Transpose the DataFrame
#     transposed_table = table.transpose()

#     # Rename the first column to use it as a header after transposition
#     transposed_table.columns = transposed_table.iloc[0]
#     transposed_table = transposed_table[1:]
    
#     transposed_table.to_csv(save_directory + 'accuracy_table.csv', index=False)
    
#     return(transposed_table)


# In[ ]:




