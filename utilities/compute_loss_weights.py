import numpy as np
from argparse import ArgumentParser


def get_dataset_name_from_lmpath(landmarks_path):
    dataset_name = landmarks_path.split('/')
    dataset_name = dataset_name[len(dataset_name)-3]
    return dataset_name


def get_vertices_from_template(template_path):
    # Load file and extract vertices
    coords = []
    with open(template_path) as f:
        for l in f:
            if l[0] == 'v' or l[0] == 'V':
                coords.append(l.replace('v ', '').replace('\n', ''))

    # Convert extracted vertices to numpy array
    vertices = []
    for p in coords:
        p = p.split(' ')
        list = []
        for i in range(0, len(p)):
            if p[i] != '' and p[i] != ' ':
                list.append(p[i])
        vertices.append(list)

    vertices = np.array(vertices, dtype='float64')

    return vertices


def get_landmarks_coordinates(landmarks_path, template):
    # Load data
    landmarks = np.squeeze(np.load(landmarks_path, allow_pickle=True))

    # Get landmarks coordinates from template vertices
    coords = []
    for i in landmarks:
        coords.append(template[i])

    coords = np.array(coords, dtype='float64')

    return coords, landmarks


def main(args):
    # Load necessary data
    template = get_vertices_from_template(args.tpath)
    landmarks, landmarks_indices = get_landmarks_coordinates(args.lmpath, template)

    print('The template file contains ' + str(template.shape[0]) + ' vertices. There are ' + str(landmarks.shape[0]) + ' landmarks.')

    # Compute the distance of every vertex of the template from all landmarks
    d = np.zeros(shape=(template.shape[0], landmarks.shape[0]), dtype='float64')
    for i in range(0, len(template)):
        for j in range(0, len(landmarks)):
            d[i][j] = np.linalg.norm(template[i] - landmarks[j])

    # Find closest landmark
    min_dist = np.zeros(shape=(len(d)), dtype='float64')
    for i in range(0, len(d)):
        m = min(d[i])
        if m == 0:
            min_dist[i] = 1e-100  # Vertices which coincide with a landmark are temporarily assigned a very small number instead of 0 (otherwise inversion returns inf)
        else:
            min_dist[i] = m

    # Invert weights
    inv_min_dist = np.zeros(shape=(len(min_dist)), dtype='float64')
    for i in range(0, len(min_dist)):
        inv_min_dist[i] = 1 / min_dist[i]

    # Scale weights
    weights = np.zeros(shape=(len(inv_min_dist)), dtype='float64')
    minimum = np.min(inv_min_dist)
    maximum = np.max(inv_min_dist)
    diff = maximum - minimum

    for i in range(0, len(inv_min_dist)):
        weights[i] = (inv_min_dist[i] - minimum) / diff

    # TODO Verifica che i vertici coincidenti coi landmarks abbiano effettivamente peso 1
    for i in range(0, 66):
        if weights[landmarks_indices[i]] != 1:
            print('Trovato peso non uguale a 1, indice:', i)

    if args.save:
        np.save(args.savepath + args.filename, weights)
        print('Loss weights have been saved to ' + args.savepath + args.filename + '.')
    else:
        print('Loss weights have not been saved.')

    return


if __name__ == '__main__':

    parser = ArgumentParser(description='Computes weights for a weighted loss; based on a series of landmark points in dense corrispondence with points on a given template.')

    parser.add_argument('lmpath', help='npy landmarks file path.')
    parser.add_argument('tpath', help='Template file path.')
    parser.add_argument('-sv', '--save', help='Save weights file (default: true).')
    parser.add_argument('-sp', '--savepath', help='Weights file path (default is ../data/dataset_name/loss_weights/).')
    parser.add_argument('-fn', '--filename', help='Weights file name (default is loss_weights.npy).')

    args = parser.parse_args()

    if args.save == 'False':
        args.save = False
    elif args.save == 'True':
        args.save = True

    if args.savepath is not None:
        if args.savepath[len(args.savepath) - 1] != '/':
            args.savepath += '/'
    else:
        dataset_name = get_dataset_name_from_lmpath(args.lmpath)
        args.savepath = '../data/' + dataset_name + '/loss_weights/'

    if args.filename is None:
        args.filename = 'loss_weights.npy'

    main(args)
