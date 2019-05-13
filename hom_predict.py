from tensorflow.keras.models import model_from_json
import hom_train
import hom_constants
import random
import matplotlib.image as mpimg
from os.path import join
import numpy as np
import time
import scipy.stats
import cv2 as cv
import hom_RANSAC

def create_dataset(path_to_images, num_examples):
    dataset = []
    orig_corners_squares = []
    perturbed_corners_squares = []

    list_images_names = hom_train.get_list_of_files(path_to_images)
    list_images_used = []

    for i in range(num_examples):
        rand_image_name = random.choice(list_images_names)

        img = mpimg.imread(join(path_to_images, rand_image_name))

        patch_orig, patch_warped, corners_square, perturbed_corners_square = hom_train.generate_crop_patches(img)
        example = hom_train.generate_homography_example(patch_orig, patch_warped, corners_square, perturbed_corners_square)

        list_images_used.append(rand_image_name)
        dataset.append(example)
        orig_corners_squares.append(corners_square)
        perturbed_corners_squares.append(perturbed_corners_square)

    return list_images_used, dataset, orig_corners_squares, perturbed_corners_squares


def compute_confidence_interval(list, confidence):
    list = np.array(list)
    num_elems = len(list)
    mean = np.mean(list)
    std_error = scipy.stats.sem(list)

    error_term = std_error * scipy.stats.t.ppf((1 + confidence) / 2.0, num_elems - 1)

    return mean, error_term

if __name__ == "__main__":
    # Load trained model
    json_file = open('hom_net_reg.json', 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("hom_net_reg.h5")
    print("Model loaded!")

    imgs_list, test_dataset, orig_corners_squares, perturbed_corners_squares = create_dataset(hom_constants.TEST_IMAGES_PATH, hom_constants.NUM_TEST_EXAMPLES)
    print("Finished generating test examples!")

    homnet_avg_error_list = []
    homnet_timing_list = []
    ransac_avg_error_list = []
    ransac_timing_list = []
    for img_name, entry, corners_square, perturbed_corners_square in zip(imgs_list, test_dataset, orig_corners_squares, perturbed_corners_squares):
        """
        Compute mean corner error and timing for HomographyNet
        """
        entry_input = entry[0]
        example_input = np.zeros((1, hom_constants.PATCH_SIZE, hom_constants.PATCH_SIZE, 2))
        example_input[0] = entry_input

        start_time = time.time()
        prediction = model.predict(example_input)
        elapsed_time = time.time() - start_time

        predicted_corners_diff = prediction.reshape(4, 2)

        predicted_perturbed_corners = np.array(corners_square) + np.array(predicted_corners_diff)
        error = predicted_perturbed_corners - np.array(perturbed_corners_square)

        avg_corner_error = np.linalg.norm(error.flatten(), ord=1) / 4.0

        homnet_avg_error_list.append(avg_corner_error)
        homnet_timing_list.append(elapsed_time)

        """
        Compute mean corner error and timing for ORB+RANSAC
        """
        # Read in the image, resize it and convert it to grayscale
        img = mpimg.imread(join(hom_constants.TEST_IMAGES_PATH, img_name))
        resized_img = cv.resize(img, (hom_constants.IMG_WIDTH, hom_constants.IMG_HEIGHT))
        gray_img = resized_img
        if len(resized_img.shape) == 3:
            gray_img = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)

        H = cv.getPerspectiveTransform(np.float32(corners_square), np.float32(perturbed_corners_square))

        warped_img = cv.warpPerspective(gray_img, H, (hom_constants.IMG_WIDTH, hom_constants.IMG_HEIGHT))

        start_time = time.time()
        estimated_H = hom_RANSAC.RANSAC_find_homography(gray_img, warped_img)
        if estimated_H is None:
            continue
        elapsed_time = time.time() - start_time

        orig_corners = np.float32(corners_square).reshape(-1, 1, 2)
        predicted_perturbed_corners = cv.perspectiveTransform(orig_corners, estimated_H)

        error = np.array(predicted_perturbed_corners).flatten() - np.array(perturbed_corners_square).flatten()

        avg_corner_error = np.linalg.norm(error.flatten(), ord=1) / 4.0

        ransac_avg_error_list.append(avg_corner_error)
        ransac_timing_list.append(elapsed_time)

    CONFIDENCE = 0.95
    homnet_mean_avg_error, homnet_pm_term_error = compute_confidence_interval(homnet_avg_error_list, CONFIDENCE)
    homnet_mean_time, homnet_pm_term_timing = compute_confidence_interval(homnet_timing_list, CONFIDENCE)

    print("HomographyNET - Mean average corner error: " + str(homnet_mean_avg_error) + " with plus/minus " + str(homnet_pm_term_error))
    print("HomographyNET - Mean timing: " + str(homnet_mean_time) + " with plus/minus " + str(homnet_pm_term_timing))

    ransac_mean_avg_error, ransac_pm_term_error = compute_confidence_interval(ransac_avg_error_list, CONFIDENCE)
    ransac_mean_time, ransac_pm_term_timing = compute_confidence_interval(ransac_timing_list, CONFIDENCE)

    print("Number of examples seen: " + str(len(ransac_avg_error_list)))
    print("ORB+RANSAC - Mean average corner error: " + str(ransac_mean_avg_error) + " with plus/minus " + str(ransac_pm_term_error))
    print("ORB+RANSAC  - Mean timing: " + str(ransac_mean_time) + " with plus/minus " + str(ransac_pm_term_timing))