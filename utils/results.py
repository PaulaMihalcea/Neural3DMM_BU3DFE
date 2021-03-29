import numpy as np
import pyrender
import matplotlib.pyplot as plt

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def my_cumsum_error(error, n_bins=1000):
    N_cnn, X_cnn = np.histogram(error, bins=n_bins)
    X_cnn = np.convolve(X_cnn, 0.5 * np.ones(2))
    X_cnn = X_cnn[1:-1]

    factor_cnn = 100.0 / error.shape[0]
    cumsum_N_cnn = np.cumsum(N_cnn)

    X_vec_cnn = np.zeros((n_bins + 1,))
    X_vec_cnn[1:] = X_cnn

    yVec_cnn = np.zeros((n_bins + 1,))
    yVec_cnn[1:] = factor_cnn * cumsum_N_cnn

    return X_vec_cnn, yVec_cnn


def save_image_face(facedata, vec, name="face", path=""):
    plt.clf()
    plt.close()

    # Generate mesh
    predict_trimeshh = facedata.vec2meshTrimesh2(vec)
    trimeshh = pyrender.Mesh.from_trimesh(predict_trimeshh, smooth=False)

    # Create scene for rendering
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
    scene.add(trimeshh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))
    camera_pose = np.array([
        [0.94063, 0.01737, -0.41513, -88.15790],
        [-0.06728, 0.98841, -0.16663, -35.36127],
        [0.33266, 0.15078, 1.14014, 241.71166],
        [0.00000, 0.00000, 0.00000, 1.00000]
    ])
    scene.add(camera, pose=camera_pose)

    # Use this in order to visualize the mesh live:
    # view = pyrender.Viewer(scene)
    # cam = view.get_my_camera_node_viewer()  # Allow access to procted attributes
    # print(view.scene.get_pose(scene.main_camera_node))

    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)
    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    img_name = path + name + ".png"
    plt.savefig(img_name)
    plt.clf()


def error_vertexs(vertices_test, predictions, std, mean):

    # Vertex error
    cnn_outputs = predictions
    # cnn_outputs = cnn_outputs[:, :-1:]
    cnn_vertices = (cnn_outputs * std) + mean
    cnn_vertices = cnn_vertices * 1000
    test_vertices = (vertices_test * std) + mean
    test_vertices = test_vertices * 1000
    errors = np.sqrt(np.sum((cnn_vertices - test_vertices) ** 2, axis=2))
    # errors.shape = (num_faces, num_vertices)  # Error for each mesh, for each vertex

    return errors


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def save_image_face_heatmap(facedata, vec, errors, id, name="face_heat", path=""):
    plt.clf()
    plt.close()
    colors_heat = []

    # Map errors to RGB colors
    min_error = 0
    max_error = 6
    for i, er in enumerate(errors[id]):
        c_er = abs(er)
        if c_er > max_error:
            c_er = max_error
        c = rgb(min_error, max_error, c_er)
        colors_heat.append(c)

    # Generate colored mesh
    predict_trimeshh = facedata.vec2meshTrimesh2(vec, col=colors_heat)
    trimeshh = pyrender.Mesh.from_trimesh(predict_trimeshh, smooth=False)

    # Create scene for rendering, etc.
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[255, 255, 255])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
    scene.add(trimeshh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    # non chiedermi mai come ho trovato questi valori bellissimi (T/N: just use these values)
    camera_pose = np.array([
        [0.94063, 0.01737, -0.41513, -88.15790],
        [-0.06728, 0.98841, -0.16663, -35.36127],
        [0.33266, 0.15078, 1.14014, 241.71166],
        [0.00000, 0.00000, 0.00000, 1.00000]
    ])
    scene.add(camera, pose=camera_pose)
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)
    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    name_image = path+name+str(".png")
    # plt.colorbar()  # Could be a nice addition, however unneeded
    plt.savefig(name_image)
    plt.clf()


def save_cumulative_distribution(errors, name, path):
    plt.clf()
    plt.cla()
    cnn_err = errors
    cnn_err = np.reshape(cnn_err, (-1,))

    X_vec_cnn, yVec_cnn = my_cumsum_error(cnn_err)

    plt.plot(X_vec_cnn, yVec_cnn, label="Mesh Autoencoder")
    plt.ylabel('Percentage of Vertices')
    plt.xlabel('Euclidean Error norm (mm)')

    plt.legend(loc='lower right')
    x_lim = 25
    plt.xlim(0, x_lim)
    plt.grid(True)  # ,color='grey', linestyle='-', linewidth=0.5)
    name = path + name + ".png"

    plt.savefig(name, bbox_inches='tight')
    name_x = path + 'x_vec_cnn.npy'
    name_y = path + 'y_vec_cnn.npy'
    with open(name_x, 'wb') as fx:
        np.save(fx, X_vec_cnn)
    with open(name_y, 'wb') as fy:
        np.save(fy, yVec_cnn)
