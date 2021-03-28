import torch
import numpy as np
from scipy.io import savemat
from argparse import ArgumentParser, Namespace as args
from facemesh import FaceData
from results import save_image_face, save_image_face_heatmap, save_cumulative_distribution
from utilities import compute_loss_weights, get_npy_from_mat, get_dataset_split, settings_parser


def main(args):

    ##### PREPROCESSING #####

    # Get settings
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')
    settings_model = settings_parser.get_settings('Model')

    # Extract landmarks; they will be available at ./data/dataset_name/template/landmarks.npy
    if settings_system['get_landmarks'] == 'True':
        get_npy_from_mat.main(args(path='./data/' + settings_dataset['dataset_type'] + '/src/' + settings_dataset['landmarks_mat'], section=settings_dataset['landmarks_mat'].replace('.mat', ''), print=False, save=True, savepath='./../data/BU3DFE/template', filename='landmarks'))  # Uncomment for first time execution

    # Compute weights; they will be available at ./data/dataset_name/loss_weights/loss_weights.npy
    if settings_system['get_weights'] == 'True':
        compute_loss_weights.main(args(lmpath='./data/' + settings_dataset['dataset_type'] + '/template/landmarks.npy', tpath='./data/' + settings_dataset['dataset_type'] + '/template/template.obj', save=True, savepath=None, filename=None, plot=True))  # Uncomment for first time execution

    # Split dataset; the splits will be available at ./data/dataset_name/preprocessed/train.npy and ./data/dataset_name/preprocessed/test.npy
    if settings_system['split_dataset'] == 'True':
        get_dataset_split.main(args(path='./data/' + settings_dataset['dataset_type'] + '/dataset.npy', save=True, split_type=settings_dataset['split_type'], split_args=float(settings_dataset['split_args']), shuffle=settings_dataset['shuffle_dataset'] == 'True', seed=int(settings_dataset['dataset_seed'] == 'True')))


    ##### GPU #####
    # GPU
    if settings_system['gpu'] == 'True':
        GPU = True
        device_idx = int(settings_system['gpu_device_idx'])  # 0
        torch.cuda.get_device_name(device_idx)
    elif settings_system['gpu'] == 'False':
        GPU = False


    ##### MAIN #####

    import json
    import os
    import copy
    import pickle

    import mesh_sampling
    import trimesh
    from shape_data import ShapeData

    from autoencoder_dataset import autoencoder_dataset
    from torch.utils.data import DataLoader

    from spiral_utils import get_adj_trigs, generate_spirals
    from models import SpiralAutoencoder
    from train_funcs import train_autoencoder_dataloader
    from test_funcs import test_autoencoder_dataloader

    from tensorboardX import SummaryWriter

    from sklearn.metrics.pairwise import euclidean_distances

    meshpackage = settings_model['meshpackage']  # mpi-mesh, trimesh
    root_dir = settings_dataset['dataset_path']  # /path/to/dataset/root_dir

    dataset = settings_dataset['dataset_type']  # COMA, DFAUST, BU3DFE
    name = ''

    ######################################################################

    args = {}

    generative_model = 'autoencoder'
    downsample_method = settings_model['downsample_method']  # COMA_downsample, BU3DFE_downsample (identical to COMA_downsample), meshlab_downsample

    # below are the arguments for the DFAUST run
    reference_mesh_file = os.path.join(root_dir, dataset, 'template', 'template.obj')
    downsample_directory = os.path.join(root_dir, dataset, 'template', downsample_method)
    ds_factors = [4, 4, 4, 4]
    step_sizes = [2, 2, 1, 1, 1]
    filter_sizes_enc = [[3, 16, 32, 64, 128], [[],[],[],[],[]]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[],[],[],[],3]]
    dilation_flag = True
    if dilation_flag:
        dilation=[2, 2, 1, 1, 1]
    else:
        dilation = None
    reference_points = [[414]]  # [[414]]; [[3567,4051,4597]] used for COMA with 3 disconnected components

    args = {'generative_model': generative_model,
            'name': name, 'data': os.path.join(root_dir, dataset, 'preprocessed',name),
            'results_folder':  os.path.join(root_dir, dataset,'results/spirals_'+ generative_model),
            'reference_mesh_file':reference_mesh_file, 'downsample_directory': downsample_directory,
            'checkpoint_file': 'checkpoint',

            'seed': int(settings_model['seed']), 'loss': settings_model['loss'],
            'batch_size': int(settings_model['batch_size']), 'num_epochs': int(settings_model['epochs']), 'eval_frequency': int(settings_model['eval_frequency']), 'num_workers': int(settings_model['num_workers']),
            'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
            'nz': int(settings_model['latent_vector']),
            'ds_factors': ds_factors, 'step_sizes' : step_sizes, 'dilation': dilation,  # seed: 2, loss: l1, batch_size: 16, num_epochs: 300, eval_frequency: 200, num_workers: 4, nz: 16

            'lr': float(settings_model['learning_rate']),  # 1e-3
            'regularization': float(settings_model['regularization']),  # 5e-5
            'scheduler': settings_model['scheduler'] == 'True', 'decay_rate': float(settings_model['decay_rate']), 'decay_steps': int(settings_model['decay_steps']),  # scheduler: True, decay_rate: 0.99, decay_steps: 1
            'resume': False,

            'mode': settings_model['mode'], 'shuffle': settings_model['shuffle'] == 'True', 'nVal': int(settings_model['nval']), 'normalization': settings_model['normalization'] == 'True'}  # mode: train, shuffle: True, nVal: 100, normalization: True

    args['results_folder'] = os.path.join(args['results_folder'],'latent_'+str(args['nz'])+'_'+str(args['loss']))

    if not os.path.exists(os.path.join(args['results_folder'])):
        os.makedirs(os.path.join(args['results_folder']))

    summary_path = os.path.join(args['results_folder'],'summaries',args['name'])
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    checkpoint_path = os.path.join(args['results_folder'],'checkpoints', args['name'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    samples_path = os.path.join(args['results_folder'],'samples', args['name'])
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    prediction_path = os.path.join(args['results_folder'],'predictions', args['name'])
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if not os.path.exists(downsample_directory):
        os.makedirs(downsample_directory)

    ######################################################################

    np.random.seed(args['seed'])
    print("Loading data...")
    if not os.path.exists(args['data']+'/mean.npy') or not os.path.exists(args['data']+'/std.npy'):
        shapedata =  ShapeData(nVal=args['nVal'],
                              train_file=args['data']+'/train.npy',
                              test_file=args['data']+'/test.npy',
                              reference_mesh_file=args['reference_mesh_file'],
                              normalization = args['normalization'],
                              meshpackage = meshpackage, load_flag = True)
        np.save(args['data']+'/mean.npy', shapedata.mean)
        np.save(args['data']+'/std.npy', shapedata.std)
    else:
        shapedata = ShapeData(nVal=args['nVal'],
                             train_file=args['data']+'/train.npy',
                             test_file=args['data']+'/test.npy',
                             reference_mesh_file=args['reference_mesh_file'],
                             normalization = args['normalization'],
                             meshpackage = meshpackage, load_flag = False)
        shapedata.mean = np.load(args['data']+'/mean.npy')
        shapedata.std = np.load(args['data']+'/std.npy')
        shapedata.n_vertex = shapedata.mean.shape[0]
        shapedata.n_features = shapedata.mean.shape[1]

    if not os.path.exists(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl')):
        if shapedata.meshpackage == 'trimesh':
            raise NotImplementedError('Rerun with mpi-mesh as meshpackage')
        print("Generating Transform Matrices ..")
        if downsample_method == 'COMA_downsample' or downsample_method == 'BU3DFE_downsample':
            M,A,D,U,F = mesh_sampling.generate_transform_matrices(shapedata.reference_mesh, args['ds_factors'])
        with open(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl'), 'wb') as fp:
            M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
            pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)
    else:
        print("Loading Transform Matrices ..")
        with open(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl'), 'rb') as fp:
            #downsampling_matrices = pickle.load(fp,encoding = 'latin1')
            downsampling_matrices = pickle.load(fp)

        M_verts_faces = downsampling_matrices['M_verts_faces']
        if shapedata.meshpackage == 'mpi-mesh':
            from psbody.mesh import Mesh
            M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
        elif shapedata.meshpackage == 'trimesh':
            M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process = False) for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']

    # Needs also an extra check to enforce points to belong to different disconnected component at each hierarchy level
    print("Calculating reference points for downsampled versions..")
    for i in range(len(args['ds_factors'])):
        if shapedata.meshpackage == 'mpi-mesh':
            dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
        elif shapedata.meshpackage == 'trimesh':
            dist = euclidean_distances(M[i+1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist,axis=0).tolist())

    ######################################################################

    if shapedata.meshpackage == 'mpi-mesh':
        sizes = [x.v.shape[0] for x in M]
    elif shapedata.meshpackage == 'trimesh':
        sizes = [x.vertices.shape[0] for x in M]
    Adj, Trigs = get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage = shapedata.meshpackage)

    spirals_np, spiral_sizes,spirals = generate_spirals(args['step_sizes'],
                                                        M, Adj, Trigs,
                                                        reference_points = reference_points,
                                                        dilation = args['dilation'], random = False,
                                                        meshpackage = shapedata.meshpackage,
                                                        counter_clockwise = True)

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
        u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
        d[0,:-1,:-1] = D[i].todense()
        u[0,:-1,:-1] = U[i].todense()
        d[0,-1,-1] = 1
        u[0,-1,-1] = 1
        bD.append(d)
        bU.append(u)

    ######################################################################

    torch.manual_seed(args['seed'])

    if GPU:
        device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]
    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]

    ######################################################################

    # Building model, optimizer, and loss function

    dataset_train = autoencoder_dataset(root_dir = args['data'], points_dataset = 'train',
                                               shapedata = shapedata,
                                               normalization = args['normalization'])

    dataloader_train = DataLoader(dataset_train, batch_size=args['batch_size'],\
                                         shuffle = args['shuffle'], num_workers = args['num_workers'])

    dataset_val = autoencoder_dataset(root_dir = args['data'], points_dataset = 'val',
                                             shapedata = shapedata,
                                             normalization = args['normalization'])

    dataloader_val = DataLoader(dataset_val, batch_size=args['batch_size'],\
                                         shuffle = False, num_workers = args['num_workers'])


    dataset_test = autoencoder_dataset(root_dir = args['data'], points_dataset = 'test',
                                              shapedata = shapedata,
                                              normalization = args['normalization'])

    dataloader_test = DataLoader(dataset_test, batch_size=args['batch_size'],\
                                         shuffle = False, num_workers = args['num_workers'])



    if 'autoencoder' in args['generative_model']:
            model = SpiralAutoencoder(filters_enc = args['filter_sizes_enc'],
                                      filters_dec = args['filter_sizes_dec'],
                                      latent_size=args['nz'],
                                      sizes=sizes,
                                      spiral_sizes=spiral_sizes,
                                      spirals=tspirals,
                                      D=tD, U=tU,device=device).to(device)


    optim = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['regularization'])
    if args['scheduler']:
        scheduler=torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'],gamma=args['decay_rate'])
    else:
        scheduler = None

    if args['loss']=='l1':
        def loss_l1(outputs, targets):
            L = torch.abs(outputs - targets).mean()
            return L
        loss_fn = loss_l1

    # Weighted loss
    elif args['loss']=='wloss':
        # Convert weights array to a Torch compatible format
        def get_duplicate_weights(weights):
            weights_long = np.ones((weights.shape[0] + 1, 3))
            for i in range(0, len(weights)):
                for j in range(0, 2):
                    weights_long[i][j] = weights[i]
            for j in range(0, 2):
                weights_long[len(weights_long) - 1][j] = 0
            weights_long = torch.from_numpy(weights_long)
            return weights_long

        weights = get_duplicate_weights(np.load('./data/' + settings_dataset['dataset_type'] + '/loss_weights/loss_weights.npy')).to(device)

        def loss_l1_weighted(outputs, targets):
            L = torch.empty(targets.shape[0], targets.shape[1], targets.shape[2])
            for i in range(0, outputs.shape[0]):
                L[i] = torch.mul(torch.abs(outputs[i] - targets[i]), weights)
            L = L.mean()
            return L
        loss_fn = loss_l1_weighted

    ######################################################################

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # print(M[4].v.shape)

    ######################################################################

    if args['mode'] == 'train':
        writer = SummaryWriter(summary_path)
        with open(os.path.join(args['results_folder'],'checkpoints', args['name'] +'_params.json'),'w') as fp:
            saveparams = copy.deepcopy(args)
            json.dump(saveparams, fp)

        if args['resume']:
                print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file'])))
                checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
                start_epoch = checkpoint_dict['epoch'] + 1
                model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
                optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
                print('Resuming from epoch %s'%(str(start_epoch)))
        else:
            start_epoch = 0

        if args['generative_model'] == 'autoencoder':
            train_autoencoder_dataloader(dataloader_train, dataloader_val,
                              device, model, optim, loss_fn,
                              bsize = args['batch_size'],
                              start_epoch = start_epoch,
                              n_epochs = args['num_epochs'],
                              eval_freq = args['eval_frequency'],
                              scheduler = scheduler,
                              writer = writer,
                              save_recons=True,
                              shapedata=shapedata,
                              metadata_dir=checkpoint_path, samples_dir=samples_path,
                              checkpoint_path = args['checkpoint_file'])

    ######################################################################

    if args['mode'] == 'test':
        print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar')))
        checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

        predictions, testset, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test,
                                                                     shapedata, mm_constant = 1000)
        np.save(os.path.join(prediction_path,'predictions'), predictions)

        print('autoencoder: normalized loss', norm_l1_loss)

        print('autoencoder: euclidean distance in mm=', l2_loss)

        ######################################################################

        # Get necessary data for images and/or mat files
        if settings_system['save_images'] == 'True' or settings_system['save_mat'] == 'True':
            # Calculate the error of all faces, for all vertices
            mean = np.load(args['data'] + '/mean.npy')
            std = np.load(args['data'] + '/std.npy')
            test_vert = testset  # np.load(args['data']+'/test.npy')

            cnn_out = np.load(args['results_folder']+'/predictions/predictions.npy')
            cnn_outputs = cnn_out[:, :-1, :]
            cnn_vertices = ((cnn_outputs * std) + mean) * 1000
            test_vertices = test_vert  # * 1000  # ((test_vert * std) + mean) * 1000

            facedata = FaceData(nVal=100, train_file=args['data'] + '/train.npy', test_file=args['data'] + '/test.npy', reference_mesh_file=reference_mesh_file, pca_n_comp=8, fitpca=True)

        # Save images
        if settings_system['save_images'] == 'True':
            # Create additional folders
            # Image folder
            path_images = os.path.join(args['results_folder'] + '/images/')
            if not os.path.exists(path_images):
                os.makedirs(path_images)

            # Cumulative distribution folder
            path_cumulativ_distrs = path_images + 'cumulativ_distrs/'
            if not os.path.exists(path_cumulativ_distrs):
                os.makedirs(path_cumulativ_distrs)

            # Predicted images folder
            path_predicted = path_images + 'predicted/'
            if not os.path.exists(path_predicted):
                os.makedirs(path_predicted)

            # Vanilla predicted faces folder
            path_faces = path_predicted + 'faces/'
            if not os.path.exists(path_faces):
                os.makedirs(path_faces)

            # Heatmaps folder
            path_heatmap = path_predicted + 'heatmap/'
            if not os.path.exists(path_heatmap):
                os.makedirs(path_heatmap)

            # References folder
            path_reference = path_images + 'reference/'
            if not os.path.exists(path_reference):
                os.makedirs(path_reference)

            # Vanilla references folder
            path_faces_ref = path_reference + 'faces/'
            if not os.path.exists(path_faces_ref):
                os.makedirs(path_faces_ref)

            # Reference heatmaps folder  (not really needed)
            # path_heatmap_ref = path_reference + 'heatmap/'
            # if not os.path.exists(path_heatmap_ref):
            #     os.makedirs(path_heatmap_ref)

            errors = np.sqrt(np.sum((cnn_vertices - test_vertices) ** 2, axis=2))
            print('Mean euclidean error: ', np.mean(errors), 'max error:', np.max(errors))

            # Compute cumulative distribution
            name_distr = 'cumulativ_distr_nz_' + str(args['nz'])
            save_cumulative_distribution(errors, name_distr, path_cumulativ_distrs)

            # Compute necessary data
            num_test = predictions.shape[0]
            ids = range(0, num_test, 1)

            for id in ids:
                # Predictions
                vec = cnn_vertices[id]  # vec.shape = (5023, 3), type: nd.array
                # vec = vec[:-1]

                # Save predicted image with identity == id
                name_image = str(id) + '_face_nz' + str(str(args['nz']))
                save_image_face(facedata, vec, name_image, path_faces)

                # Save heatmap of predicted image with identity == id
                name_heat = str(id) + '_face_heat_nz' + str(str(args['nz']))
                save_image_face_heatmap(facedata, vec, errors, id, name_heat, path_heatmap)

                # Save reference test set image (for comparison)
                vec_test = test_vertices[id]
                name_reference = str(id) + '_ref_nz' + str(str(args['nz']))
                save_image_face(facedata, vec_test, name_reference, path_faces_ref)

                # Save heatmap of reference test set image (for comparison); since the error is 0, this step is actually unneeded
                # vec_test = test_vertices[id]
                # name_reference_heat = str(id) + '_heatref_nz' + str(str(args['nz']))
                # save_image_face_heatmap(facedata, vec_test, name_reference_heat, id, name_reference_heat, path_heatmap_ref)

        # Save mat files
        if settings_system['save_mat'] == 'True':
            # mat file folder
            path_mat = os.path.join(args['results_folder'] + '/mat/')
            if not os.path.exists(path_mat):
                os.makedirs(path_mat)

            # Save mat files ({'mydata': mydata})
            savemat(path_mat + 'test_faces', {'all_faces_test' : test_vertices})
            savemat(path_mat + 'recon_faces', {'all_faces_recostructed' : cnn_vertices})


######################################################################

if __name__ == '__main__':

    parser = ArgumentParser(description='Main script for the Neural3DMM project.')

    parser.add_argument('-st', '--settings', help='Choose a settings file (default is the one specified in settings/setup_file).')
    parser.add_argument('-m', '--mode', help='Choose a mode between \'train\' and \'test\' (default is the one specified in the default settings file).')

    args = parser.parse_args()

    if args.settings is not None:
        settings_parser.set_setup_file('./settings/' + args.settings + '.cfg')
    if args.mode is not None:
        settings_parser.set_settings('System', 'mode', args.mode)

    main(args)
    print('Fin.')
