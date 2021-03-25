import numpy as np
import random
from argparse import ArgumentParser


def get_shuffled_models(ids, arg, seed=None):
    # Lorenzo's improvements begin
    # Get unique identities list
    identity_f = {}
    identity_m = {}

    for i in ids:
        person = i.split('_')[0]
        if 'M' in person:
            if i.split('_')[0] in identity_m:
                identity_m[person] += 1
            else:
                identity_m[person] = 1
        elif 'F' in i.split('_')[0]:
            if i.split('_')[0] in identity_f:
                identity_f[person] += 1
            else:
                identity_f[person] = 1

    identity_m = [k for k, v in identity_m.items()]
    identity_f = [k for k, v in identity_f.items()]

    # Lorenzo's improvements end

    ''' # Old version; shuffle ok, seed not working because list(set()) does not keep the order
    # Get unique identities list
    unique_ids_f = {}
    unique_ids_m = {}
    
    for i in ids:
        if 'F' in i.split('_')[0]:
            unique_ids_f.append(i.split('_')[0])
        elif 'M' in i.split('_')[0]:
            unique_ids_m.append(i.split('_')[0])
    unique_ids_f = list(set(unique_ids_f))
    unique_ids_m = list(set(unique_ids_m))
    '''

    # Calculate number of F and M identities for test dataset (easier to balance, as opposed to train)
    nm = int(arg/2)
    if nm >= len(identity_m):
        nm = len(identity_m)
    nf = arg - nm

    # Sample nf and nm random identities
    if seed is not None:
        random.seed(seed)

    indices_f = list(range(0, len(identity_f)))
    indices_m = list(range(0, len(identity_m)))
    random.shuffle(indices_f)
    random.shuffle(indices_m)

    f_models = []
    m_models = []
    for i in range(0, nf):
        f_models.append(identity_f[indices_f[i]])
    for i in range(0, nm):
        m_models.append(identity_m[indices_m[i]])

    models = f_models + m_models

    return models


def get_ordered_models(ids, arg):
    # Eliminate model details (only keep identity gender and number)
    unique_ids = ids.copy()
    for i in range(0, len(ids)):
        unique_ids[i] = ids[i].split('_')[0]
    # Get all unique identities in order
    unique_ids = list(dict.fromkeys(unique_ids))

    # Get first arg identities (or all of them)
    if arg == 'max':
        arg = len(unique_ids)
    models = unique_ids[:arg]

    return models


def split_dataset(dataset, ids, split_type, arg, shuffle, seed=None):

    train = []
    test = []
    arg = int(arg)

    if split_type == 'x_ids':

        if shuffle:  # Get x random identities
            models = get_shuffled_models(ids, arg, seed)

        else:  # Get x identities in order (from the first one)
            models = get_ordered_models(ids, arg)

        for i in range(0, len(ids)):
            if ids[i].split('_')[0] in models:
                test.append(dataset[i])
            else:
                train.append(dataset[i])

        train = np.array(train)
        test = np.array(test)

        print('The test split will contain ' + str(len(test)) + ' models.')

    elif split_type == 'p_ids':
        unique_ids = get_ordered_models(ids, 'max')  # Contains a list of all unique identities in alternate order, with only identity gender and number (F0001, M0001, F0002, M0002...)

        # Create a list containing the number of models for each identity
        models_per_id = np.zeros(len(unique_ids), dtype='int').tolist()
        for i in range(0, len(unique_ids)):
            for id in ids:
                if unique_ids[i] in id:
                    models_per_id[i] += 1

        print('Models per identity:', models_per_id)

        models_per_id_percent = []  # Contains the percentage of models for each identity that will be inserted in the test split
        for n in models_per_id:
            models_per_id_percent.append(int(n / 100 * arg))

        print('The test split will contain ' + str(sum(models_per_id_percent)) + ' models.')

        models = unique_ids.copy()

        if shuffle:
            dataset_shuffle = []

            if seed is not None:
                random.seed(seed)

            for curr_id in unique_ids:  # Shuffle all models of each identity - NOTE: It does NOT shuffle identities (it is useless, as each identity will get a percentage of models in the test split)
                current_models = []
                for m in range(0, len(dataset)):  # Create an array of all the current identity's models
                    if curr_id in ids[m]:
                        current_models.append(dataset[m])
                random.shuffle(current_models)  # Shuffle the models of the current identity
                for i in range(0, len(current_models)):
                    dataset_shuffle.append(current_models[i])

            ''' # Use this to test the shuffle part; all numbers in the given range should correspond in a 1-to-1 relationship (there should be no i greater than the given range, i.e. 17)
            for i in range(0, len(dataset)):
                for j in range(0, 18):
                    if np.array_equal(dataset[j], dataset_shuffled[i]):
                        print('Indices:', j, i)
            '''

            for i in range(0, len(models)):  # For each identity (100 total)
                counter = 0
                for j in range(0, len(ids)):  # For each model (1779 total)
                    if ids[j].split('_')[0] in models[i]:  # If the current identity (ids[j].split('_')[0]) can be found in the list of models to be taken, then append it to the dataset (get as many models as the models needed to meet the percentage)
                        if counter < models_per_id_percent[i]:
                            test.append(dataset_shuffle[j])
                            counter += 1
                        else:
                            train.append(dataset_shuffle[j])

        # Note that this function gets the first n / 100 * arg models of each identity, in order
        # Meaning that most probably each split will have a majority of certain expressions (e.g. if the first expression of each identity is neutral, then the test split will mostly contain neutral expressions, while the train split will get the remaining others)
        else:  # Ordered version
            for i in range(0, len(models)):  # For each identity (100 total)
                counter = 0
                for j in range(0, len(ids)):  # For each model (1779 total)
                    if ids[j].split('_')[0] in models[i]:  # If the current identity (ids[j].split('_')[0]) can be found in the list of models to be taken, then append it to the dataset (get as many models as the models needed to meet the percentage)
                        if counter < models_per_id_percent[i]:
                            test.append(dataset[j])
                            counter += 1
                        else:
                            train.append(dataset[j])

    train = np.array(train)
    test = np.array(test)

    return train, test


def main(args):
    # Load dataset
    print('Loading dataset... ', end='')
    dataset = np.load(args.path, allow_pickle=True)
    print('dataset shape:', dataset.shape)

    # Load identities list
    print('Loading identities... ', end='')
    # Get identities file path
    dataset_path = ''
    d_path = args.path.split('/')
    d_path = d_path[:len(d_path)-1]
    for i in range(0, len(d_path)):
        dataset_path += d_path[i] + '/'

    with open(dataset_path + 'identities.txt', 'r') as f:
        ids = f.read().splitlines()
    ids = np.array(ids)

    print('identity array length:', len(ids))

    print('Splitting dataset with split type "' + str(args.split_type) + '", arguments "' + str(args.split_args) + '", shuffle "' + str(args.shuffle) + '" and seed ' + str(args.seed) + '...')
    train, test = split_dataset(dataset, ids, args.split_type, args.split_args, args.shuffle, int(args.seed))
    print('Done.')

    if args.save:
        np.save(dataset_path + 'preprocessed/' + 'train', train)
        np.save(dataset_path + 'preprocessed/' + 'test', test)
        print('Split dataset saved to ' + dataset_path + 'preprocessed/' + 'train.npy' + 'and' + dataset_path + 'preprocessed/' + 'test.npy.')

    return


if __name__ == '__main__':

    parser = ArgumentParser(description='Dataset split utility.')

    parser.add_argument('path', help='Dataset path (npy file).')
    parser.add_argument('save', help='Save split.')
    parser.add_argument('split_type', help='Split type (x_ids, p_ids).')
    parser.add_argument('split_args', type=float, help='Split type argument (x_ids: number of ids for test split; p_ids: percentage of models per id for test split).')
    parser.add_argument('shuffle', help='Shuffle dataset (True/False).')
    parser.add_argument('-seed', '--seed', help='Seed for random shuffle (optional).')
    parser.add_argument('-fn', '--filename', help='Split folder filename.')

    args = parser.parse_args()

    if args.save is None or args.save == 'False':
        args.save = False
    elif args.save == 'True':
        args.save = True

    if args.shuffle is None or args.shuffle == 'False':
        args.shuffle = False
    elif args.shuffle == 'True':
        args.shuffle = True

    main(args)
