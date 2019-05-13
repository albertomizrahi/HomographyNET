import cv2 as cv
import numpy as np

def RANSAC_find_homography(img, warped_img):
    """
    The following code is based off of the one found here:
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    """

    # Initialize the SIFT detector
    orb_detector = cv.ORB_create()

    # Find the SIFT keypoints and descriptors from both images
    keypoints1, descriptors1 = orb_detector.detectAndCompute(img, None)
    keypoints2, descriptors2 = orb_detector.detectAndCompute(warped_img, None)

    # Use the FLANN (Fast Library for Approx. Nearest Neighbors) to find the best matches
    # between the descriptors of both images
    # Set up the parameters for FLANN
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=100)

    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)

    flann_matches = flann_matcher.knnMatch(descriptors1, descriptors2, 2)

    # Store all the matches that are good, were good is defined as passing the Lowe's
    # ratio test
    good_matches = []
    for matches in flann_matches:
        if len(matches) != 2:
            continue
        d1, d2 = matches
        if d1.distance < 0.7 * d2.distance:
            good_matches.append(d1)

    num_good_matches = len(good_matches)
    if num_good_matches < 5:
        print("ERROR: not enough good matches for estimating homography")
        return None

    # Extract locations of the matched keypoints
    img_points = [keypoints1[m.queryIdx].pt for m in good_matches]
    warped_img_points = [keypoints2[m.trainIdx].pt for m in good_matches]

    # Reshaped the lists so they can be passed to findHomography()
    img_points = np.float32(img_points).reshape(-1, 1, 2)
    warped_img_points = np.float32(warped_img_points).reshape(-1, 1, 2)

    H, _ = cv.findHomography(img_points, warped_img_points, cv.RANSAC, ransacReprojThreshold=5.0)

    return H

if __name__ == "__main__":
    pass
