import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPooling2D, Flatten
import keras.backend as K
from os import listdir
from os.path import isfile, join
import math
import pickle
import hom_constants

def l2_loss(y_true, y_pred):
    """
    Partly taken from: https://riptutorial.com/keras/example/32022/euclidean-distance-loss
    """
    loss_squared = K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)
    """
    Before taking the sqrt of the loss squared, we need to ensure that the loss squared is not
    zero. Thus, we ensure that is bigger than some epsilon value.
    This is done to avoid issues when calculating the gradient of the sqrt function
    for backprop. 
    """
    loss = K.sqrt(K.maximum(loss_squared, K.epsilon()))
    return loss

def generate_crop_patches(img):
    # Step 1: Resize image to 320x240
    resized_img = cv.resize(img, (hom_constants.IMG_WIDTH, hom_constants.IMG_HEIGHT))


    # Step 2: Convert to grayscale
    gray_img = resized_img
    if len(resized_img.shape) == 3:
        gray_img = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)


    # Step 3: Randomly crop a square patch from the image
    top_left_x = random.randint(hom_constants.PERTURBATION_MAX, hom_constants.IMG_WIDTH - hom_constants.PATCH_SIZE - hom_constants.PERTURBATION_MAX)
    top_left_y = random.randint(hom_constants.PERTURBATION_MAX, hom_constants.IMG_HEIGHT - hom_constants.PATCH_SIZE - hom_constants.PERTURBATION_MAX)

    top_left = [top_left_x, top_left_y]
    top_right = [top_left[0] + hom_constants.PATCH_SIZE, top_left[1]]
    bottom_right = [top_left[0] + hom_constants.PATCH_SIZE, top_left[1] + hom_constants.PATCH_SIZE]
    bottom_left = [top_left[0], top_left[1] + hom_constants.PATCH_SIZE]

    patch_orig = gray_img[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]

    corners_square = [top_left, top_right, bottom_right, bottom_left]


    # Step 4: Perturb the square's corners
    perturbed_corners = []

    for corner in corners_square:
        # The +-1 in the random perturbations is to avoid generating a perturbation that will move the patch to
        # the border to prevent bordering artifacts later in the data generation pipeline
        new_x = corner[0] + random.randint(-hom_constants.PERTURBATION_MAX + 1, hom_constants.PERTURBATION_MAX - 1)
        new_y = corner[1] + random.randint(-hom_constants.PERTURBATION_MAX + 1, hom_constants.PERTURBATION_MAX - 1)
        perturbed_corners.append([new_x, new_y])

    perturbed_top_left = perturbed_corners[0]
    perturbed_top_right = perturbed_corners[1]
    perturbed_bottom_right = perturbed_corners[2]
    perturbed_bottom_left = perturbed_corners[3]

    perturbed_corners_square = [perturbed_top_left, perturbed_top_right, perturbed_bottom_right, perturbed_bottom_left]


    # Step 5: Calculate the homography H_ab
    # See here for details: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    H_ab = cv.getPerspectiveTransform(np.float32(corners_square), np.float32(perturbed_corners))

    warped_img = cv.warpPerspective(gray_img, H_ab, (hom_constants.IMG_WIDTH, hom_constants.IMG_HEIGHT))

    # Step 6: Calculate the inverse homoraphy H_ba
    H_ba = np.linalg.inv(H_ab)


    # Step 7: Create the new image I'
    warped_img = cv.warpPerspective(gray_img, H_ba, (hom_constants.IMG_WIDTH, hom_constants.IMG_HEIGHT))


    # Step 8: Crop second patch
    patch_warped = warped_img[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]

    return patch_orig, patch_warped, corners_square, perturbed_corners_square

def generate_homography_example(patch_orig, patch_warped, corners_square, perturbed_corners_square):
    # Steps 1-8:
    # Step 9: Stack the patches together depth-wise
    example = np.zeros((hom_constants.PATCH_SIZE, hom_constants.PATCH_SIZE, 2))
    # print(example.shape)
    example[:, :, 0] = patch_orig
    example[:, :, 1] = patch_warped

    corners_square = np.array(corners_square)
    perturbed_corners_square = np.array(perturbed_corners_square)

    # print(corners_square)
    # print(perturbed_corners_square)

    # Step 10: calculate the difference between the corners of the perturbed patch and the original patch
    # This is the target for the above training example
    prediction = perturbed_corners_square - corners_square
    # print(prediction)

    return example, prediction.flatten()

def get_list_of_files(directory_path):
    list_dir_files = listdir(directory_path)
    list_files = []

    for f in list_dir_files:
        if isfile(join(directory_path, f)):
            list_files.append(f)

    return list_files


# def create_dataset(path_to_images, hom_constants.NUM_EXAMPLES):
#     dataset = []
#
#     list_images_names = get_list_of_files(path_to_images)
#
#     rand_images_list = random.sample(list_images_names, hom_constants.NUM_EXAMPLES)
#
#     for i, rand_img_name in enumerate(rand_images_list):
#         if i % 1000 == 0:
#             print("Generating example " + str(i))
#
#         img = mpimg.imread(join(path_to_images, rand_img_name))
#
#         patch_orig, patch_warped, corners_square, perturbed_corners_square = generate_crop_patches(img)
#         example = generate_homography_example(patch_orig, patch_warped, corners_square, perturbed_corners_square)
#
#         dataset.append(example)
#
#     return rand_images_list, dataset

def create_dataset(path_to_images, num_examples):
    dataset = []

    list_images_names = get_list_of_files(path_to_images)
    list_images_used = []

    for i in range(num_examples):
        if i % 1000 == 0:
            print("Generating example " + str(i))

        rand_image_name = random.choice(list_images_names)

        img = mpimg.imread(join(path_to_images, rand_image_name))

        patch_orig, patch_warped, corners_square, perturbed_corners_square = generate_crop_patches(img)
        example = generate_homography_example(patch_orig, patch_warped, corners_square, perturbed_corners_square)

        list_images_used.append(rand_image_name)
        dataset.append(example)

    return list_images_used, dataset

def prepare_data_generator(dataset):
    hom_constants.NUM_EXAMPLES = len(dataset)

    random.shuffle(dataset)

    current_example_id = 0

    while True:
        batch_example_pairs = dataset[current_example_id: current_example_id+hom_constants.BATCH_SIZE]
        current_example_id += hom_constants.BATCH_SIZE

        if current_example_id >= hom_constants.NUM_EXAMPLES:
            current_example_id = 0
            random.shuffle(dataset)

        batch_examples = []
        batch_predictions = []

        for example, prediction in batch_example_pairs:
            batch_examples.append(example)
            batch_predictions.append(prediction)

        yield np.array(batch_examples), np.array(batch_predictions)


if __name__ == "__main__":
    input_shape = (hom_constants.PATCH_SIZE, hom_constants.PATCH_SIZE, 2)

    model = Sequential()

    # Convolution layer 1
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    # Convolution layer 2
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution layer 3
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    # Convolution layer 4
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution layer 5
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    # Convolution layer 6
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution layer 7
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    # Convolution layer 8
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(8))

    print(model.summary())

    model.compile(optimizer='adam', loss=l2_loss)

    # Generate training dataset
    imgs_list, train_dataset = create_dataset(hom_constants.TRAIN_IMAGES_PATH, hom_constants.NUM_EXAMPLES)

    generator = prepare_data_generator(train_dataset)

    model.fit_generator(generator, steps_per_epoch=math.ceil(hom_constants.NUM_EXAMPLES / hom_constants.BATCH_SIZE), epochs=hom_constants.EPOCHS)

    # Serialize model to JSON
    model_json = model.to_json()
    with open("hom_net_reg.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights("hom_net_reg.h5")

    # Saved some of the training examples for visualization purposes
    with open('examples_visualization.pickle', 'wb') as f:
        pickle.dump(imgs_list, f)

    print("Saved model to disk")