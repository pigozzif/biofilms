import os

import cv2
import numpy as np
from sklearn.neighbors import KDTree
import networkx as nx
import pandas as pd
from skimage.measure import regionprops, label

NUMBER_OF_CELLS = 90
VIDEO_LENGTH = 45
FRAME_SHAPE = (1120 * 2, 850)


def read_video(file_name):
    images = []
    cap = cv2.VideoCapture(file_name)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if i % 1 == 0:
                if frame.shape[0] != FRAME_SHAPE[1] or frame.shape[1] != int(FRAME_SHAPE[0] / 2):
                    frame = cv2.resize(frame, (FRAME_SHAPE[1], int(FRAME_SHAPE[0] / 2)))
                images.append(frame)
        else:
            break
        i += 1
    cap.release()
    return images


def write_video(title, images=None, side_images=None):
    n = len(os.listdir("frames")) if images is None else len(images)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(title, fourcc, int(n / VIDEO_LENGTH), FRAME_SHAPE)
    for i in range(n):
        frame = cv2.imread("frames/{}.png".format(i)) if images is None else images[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if side_images is not None:
            frame = cv2.hconcat([frame, side_images[i]])
        out.write(frame)
    out.release()


def binarize_image(image, threshold, return_original=False):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, im_th = cv2.threshold(im_gray, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image=im_th, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(x) > 2.0, contours))
    if return_original:
        return im_th, contours
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1,
                     lineType=cv2.LINE_AA)
    return image_copy, contours


def center_of_contour(c):
    m = cv2.moments(c)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def track_spikes(new_image, x, indices, distances, contours, data, max_dist=50.0):
    d = {"cell_{}".format(i): 0 for i in range(NUMBER_OF_CELLS)}
    for num, (idx, dist) in enumerate(zip(indices, distances)):
        i = idx[0]
        if dist[0] > max_dist:
            x = np.append(x, center_of_contour(contours[num])).reshape(-1, 2)
            i = len(x) - 1
        else:
            x[i] = center_of_contour(contours[num])
        cv2.putText(new_image, text=str(i), org=center_of_contour(contours[num]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        d["cell_{}".format(i)] = 1
    d["t"] = 676 if data.empty else data["t"].max() + 1
    data = data.append(d, ignore_index=True)
    return new_image, x, data


def build_graph(x, max_dist=175.0):
    g = nx.Graph()
    for i, node in enumerate(x):
        g.add_node(i, x=node[0], y=node[1], l=0.0, fires=False)
    distances, indices = KDTree(x, metric="euclidean").query(x, k=len(x), return_distance=True)
    for i in range(len(x)):
        for idx, dist in zip(indices[i], distances[i]):
            if idx != i and dist <= max_dist and not g.has_edge(i, idx):
                g.add_edge(i, idx)
    return g


def draw_graph(g, image):
    for u, v in g.edges:
        cv2.line(image, pt1=(g.nodes[u]["x"], g.nodes[u]["y"]), pt2=(g.nodes[v]["x"], g.nodes[v]["y"]),
                 color=(255, 0, 0), thickness=2)
    for node in g.nodes:
        cv2.circle(image, center=(g.nodes[node]["x"], g.nodes[node]["y"]), radius=5, color=(0, 0, 255), thickness=2)
    cv2.imwrite("graph.png", image)


def save_graph(g):
    nx.write_gexf(g, "graph.gexf")


def save_data(data):
    data.to_csv("spikes2.csv", sep=",", index=False)


def center_of_connected_component(labels, label):
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == label] = 255
    x, y, *_ = cv2.boundingRect(mask)
    return np.array([x, y])


def annotate_connected_component(new_image, edges, labels, label):
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == label] = 255
    x, y, w, h = cv2.boundingRect(mask)
    crop = edges.copy()
    crop[labels != label] = 0
    contours, _ = cv2.findContours(image=crop, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        new_image = cv2.rectangle(new_image, (x, y), (x + w, y + h), (138, 223, 178), 2)
        # new_image = cv2.putText(new_image, text=str(len(contours) - 1), org=(int(x + w / 2), int(y + h / 2)),
        #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(445, 160, 51), thickness=2)
        # for cnt in contours:
        #     cx, cy = center_of_contour(cnt)
        #     new_image = cv2.circle(new_image, center=(cx, cy), radius=5, color=(227, 206, 166), thickness=1)
    return new_image


def count_number_of_cells(labels, label, edges):
    crop = edges.copy()
    crop[labels != label] = 0
    contours, _ = cv2.findContours(image=crop, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return len(contours)


def detect_cells(image):
    binary, contours = binarize_image(image=image, return_original=True, threshold=70)
    new_image = image.copy()
    temp_image = cv2.copyTo(image, binary)
    temp_image = cv2.GaussianBlur(temp_image, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=temp_image, threshold1=0, threshold2=50)
    kernel = np.ones((4, 4), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    ret, labels = cv2.connectedComponents(binary)
    for label in range(1, ret):
        new_image = annotate_connected_component(new_image=new_image, edges=edges, labels=labels, label=label)
    return new_image, ret, labels, edges


def get_xenobits_updated_positions(d):
    x = np.empty(0)
    for i, traj in d.groupby(["id"]):
        traj = traj[traj["t"] == traj["t"].max()]
        x = np.append(x, traj[["x", "y"]].values).reshape(-1, 2)
    return x


def process_xenobits(images, max_dist=50.0):
    data = pd.DataFrame(columns=["t", "id", "x", "y", "area", "perimeter", "short", "long", "orientation",
                                 "solidity", "elongation", "eccentricity", "n_cells", "n_spikes"])
    x = np.empty((0, 0))
    os.makedirs("frames", exist_ok=True)
    for t, image in enumerate(images):
        print(t)
        file_name = "{}.png".format(t)
        if file_name in os.listdir("frames"):
            os.system("rm {}".format(os.path.join("frames", file_name)))
        new_image, ret, labels, edges = detect_cells(image=image)
        centers = np.array([center_of_connected_component(labels=labels, label=label) for label in range(1, ret)])\
            .reshape(-1, 2)
        if not x.size:
            x = centers.copy()
        distances, indices = KDTree(x, metric="euclidean").query(centers, k=2, return_distance=True)
        for label in range(1, ret):
            crop = image.copy()
            crop[labels != label] = 0
            _, contours = binarize_image(image=crop, threshold=210)
            i = indices[label - 1][0]
            if distances[label - 1][0] > max_dist:
                x = np.append(x, centers[label - 1]).reshape(-1, 2)
                i = len(x) - 1
            else:
                x[i] = centers[label - 1]
            crop[labels == label] = 1
            region = regionprops(crop[:, :, 0])[0]
            area = region.area
            w, h = region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]
            if region.major_axis_length <= 0.0:
                continue
            data = data.append({"t": t, "id": i, "x": centers[label - 1][0], "y": centers[label - 1][1],
                                "area": area, "perimeter": region.perimeter,
                                "short": min(w, h), "long": max(w, h),
                                "orientation": region.orientation, "solidity": region.solidity,
                                "elongation": region.minor_axis_length / region.major_axis_length,
                                "eccentricity": region.eccentricity,
                                "n_cells": count_number_of_cells(labels=labels, label=label, edges=edges),
                                "n_spikes": len(contours)},
                               ignore_index=True)
        cv2.imwrite(os.path.join("frames", file_name), new_image)
        data.to_csv("xenobits.csv", sep=",", index=False)


def process_spikes(images):
    data = pd.DataFrame(columns=["t"] + ["cell_{}".format(i) for i in range(NUMBER_OF_CELLS)])
    x = np.empty((0, 0))
    os.makedirs("frames", exist_ok=True)
    for i, image in enumerate(images):
        file_name = "{}.png".format(i)
        if file_name in os.listdir("frames"):
            continue
        new_image, contours = binarize_image(image=image, threshold=210)
        if not x.size:
            x = np.array([center_of_contour(c) for c in contours]).reshape(-1, 2)
        distances, indices = KDTree(x, metric="euclidean").query(
            np.array([center_of_contour(c) for c in contours]).reshape(-1, 2), k=2, return_distance=True)
        new_image, x, data = track_spikes(new_image=new_image, x=x, indices=indices, distances=distances,
                                          contours=contours, data=data)
        cv2.imwrite(os.path.join("frames", file_name), new_image)
    # save_data(data=data)
    # np.save("x.npy", x)
    # g = build_graph(x=x)
    # save_graph(g=g)
    # draw_graph(g=g, image=new_images[0].copy())


def process_world(images, threshold):
    os.makedirs("frames", exist_ok=True)
    for i, image in enumerate(images):
        file_name = "{}.png".format(i)
        if file_name in os.listdir("frames"):
            os.system("rm {}".format(os.path.join("frames", file_name)))
        new_image, *_ = detect_cells(image=image)
        cv2.imwrite(os.path.join("frames", file_name), new_image)


def describe_features(images):
    image = images[0]
    new_image, ret, labels, edges = detect_cells(image=image)
    largest_component = max(list(range(1, ret)), key=lambda x: np.sum(labels[labels == x]))
    crop = image.copy()
    crop[labels != largest_component] = 0
    crop[labels == largest_component] = 1
    contours, _ = cv2.findContours(image=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), mode=cv2.RETR_TREE,
                                   method=cv2.CHAIN_APPROX_NONE)
    crop[labels == largest_component] = (0, 0, 255)
    image = cv2.addWeighted(image, 0.5, crop, 0.5, 0.0)
    image_2 = image.copy()
    image_3 = image.copy()
    cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    image = annotate_connected_component(new_image=image, edges=edges, labels=labels, label=largest_component)
    cv2.putText(image, text="short", org=(30 + 120, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(138, 223, 178), thickness=2)
    cv2.putText(image, text="long", org=(170 + 135, 284 + 125), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(138, 223, 178), thickness=2)
    cv2.putText(image, text="area", org=(20 + 100, 240 + 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 0, 255), thickness=2)
    cv2.putText(image, text="perimeter", org=(120 + 135, 330 + 125), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=2)
    cv2.putText(image, text="n. cells", org=(120 + 135, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(227, 206, 166), thickness=2)
    cv2.imwrite("features1.png", image)
    ellipse = cv2.fitEllipse(contours[0])
    cv2.ellipse(image_2, ellipse, (0, 0, 255), 2)
    cv2.putText(image_2, text="minor", org=(20 + 100, 240 + 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=2)
    cv2.putText(image_2, text="major", org=(120 + 135, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=2)
    cv2.imwrite("features2.png", image_2)
    hull = cv2.convexHull(contours[0])
    cv2.drawContours(image=image_3, contours=[hull], contourIdx=-1, color=(255, 0, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    cv2.putText(image_3, text="convex hull", org=(120 + 135, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=2)
    cv2.imwrite("features3.png", image_3)


def main(file_name):
    images = read_video(file_name=file_name)
    # process_world(images=images, threshold=210 if "contours" not in file_name else 125)
    # process_spikes(images=images)
    # process_xenobits(images=images)
    describe_features(images=images)
    # write_video(title="spikes_temp.mp4", side_images=images)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # data1 = pd.read_csv("spikes1.csv", sep=",")
    # data2 = pd.read_csv("spikes2.csv", sep=",")
    # data = pd.concat([data1, data2])
    # data.to_csv("spikes.csv", sep=",", index=False)
    main("../xenobits.mp4")
