import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import os


def calc_color_histogram(img, divide_img=True):

    if divide_img:
        # split the image in 3 subimages
        step = int(np.floor(img.shape[0] / 3))
        sub_images = [img[sub_id * step: (sub_id + 1) * step] for sub_id in range(3)]
        img_hist = np.zeros((3, 3 * 256))
    else:
        sub_images = [img]
        img_hist = np.zeros((1, 3 * 256))

    colors = ("b", "g", "r")
    img_meanBGR = []
    for id, sub_img in enumerate(sub_images):

        sub_img_hist = []
        # loop over the image channels
        for ch_id, color in enumerate(colors):
            hist = cv2.calcHist([sub_img], [ch_id], None, [256], [0, 256])
            hist /= hist.sum()
            # plt.plot(hist, color=color)
            sub_img_hist.extend(hist)

        img_hist[id, :] = np.array(sub_img_hist).reshape(1, -1)
        img_meanBGR.extend([np.mean(sub_img[:, :, 0]), np.mean(sub_img[:, :, 1]), np.mean(sub_img[:, :, 2])])

    return img_hist.flatten(), np.array(img_meanBGR)


def calc_color_metric(images):

    divide_img = True
    all_hists, all_mean_BGR = [], []
    all_hists_f, all_mean_BGR_f = [], []
    all_hists_lab, all_mean_BGR_lab = [], []
    for img in images:
        cv2.imshow('test', img)
        cv2.waitKey(1)

        # 1. Calculate histograms in BGR space
        img_hist, img_meanBGR = calc_color_histogram(img, divide_img)
        all_hists.append(img_hist)
        all_mean_BGR.append(img_meanBGR)

        # 2. Subtract the mean from each channel and do the same for the filtered image.
        filtered_img = img.copy()
        filtered_img[:, :, 0] = (img[:, :, 0] - np.mean(img[:, :, 0])).astype('uint8')
        filtered_img[:, :, 1] = (img[:, :, 1] - np.mean(img[:, :, 1])).astype('uint8')
        filtered_img[:, :, 2] = (img[:, :, 2] - np.mean(img[:, :, 2])).astype('uint8')

        img_hist_f, img_meanBGR_f = calc_color_histogram(filtered_img, divide_img)
        all_hists_f.append(img_hist_f)
        all_mean_BGR_f.append(img_meanBGR_f)

        # 3. Do the same in lab space
        imglab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_hist_lab, img_meanBGR_lab = calc_color_histogram(imglab, divide_img)
        all_hists_lab.append(img_hist_lab)
        all_mean_BGR_lab.append(img_meanBGR_lab)

    # Check the similarity/distance of the first image with the rest.
    hist1 = all_hists[0]
    meanBGR_1 = all_mean_BGR[0]
    hist1_f = all_hists_f[0]
    meanBGR_1_f = all_mean_BGR_f[0]
    hist1_lab = all_hists_lab[0]
    meanBGR_1_lab = all_mean_BGR_lab[0]
    for i in range(len(images)):
        hist2 = all_hists[i]
        cos_sim = np.dot(hist1.T, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
        #cos_sim = np.sum(hist1 * hist2, axis=1) / (np.linalg.norm(hist1, axis=1) * np.linalg.norm(hist2, axis=1))
        meanBGR_2 = all_mean_BGR[i]
        dist = np.linalg.norm(meanBGR_1 - meanBGR_2)
        print(f'Image {i}: \nSimilarity: {cos_sim} Euclidean distance {i}: {dist}')

        hist2_f = all_hists_f[i]
        cos_sim = np.dot(hist1_f.T, hist2_f) / (np.linalg.norm(hist1_f) * np.linalg.norm(hist2_f))
        meanBGR_2_f = all_mean_BGR_f[i]
        dist = np.linalg.norm(meanBGR_1_f - meanBGR_2_f)
        print(f'Similarity: {cos_sim} Euclidean distance {i}: {dist}')

        hist2_lab = all_hists_lab[i]
        cos_sim = np.dot(hist1_lab.T, hist2_lab) / (np.linalg.norm(hist1_lab) * np.linalg.norm(hist2_lab))
        meanBGR_2_lab = all_mean_BGR_lab[i]
        dist = np.linalg.norm(meanBGR_1_lab - meanBGR_2_lab)
        print(f'Similarity: {cos_sim} Euclidean distance {i}: {dist}')


    print('Done')


def main():
    video_dir: Path = Path("images/test/")

    img_files_y = sorted([vpath for vpath in  Path(os.path.join(video_dir, "yellow")).glob('*.*')])
    img_files_w = sorted([vpath for vpath in Path(os.path.join(video_dir, "white")).glob('*.*')])
    img_files_o = sorted([vpath for vpath in Path(os.path.join(video_dir, "other")).glob('*.*')])

    images = []
    for img_path in img_files_y:
        img = cv2.imread(str(img_path))
        images.append(img)

    for img_path in img_files_w:
        img = cv2.imread(str(img_path))
        images.append(img)

    calc_color_metric(images)


    print('Done')

if __name__ == "__main__":
    main()