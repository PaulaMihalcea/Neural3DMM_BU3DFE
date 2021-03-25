import sys
import numpy as np
from argparse import ArgumentParser
from scipy import io


def main(args):

    mat = io.loadmat(args.path)  # Read mat file

    if args.section is not None:
        # Conversion
        if isinstance(mat[args.section], (np.ndarray, np.generic)):  # Check if the specified section is a NumPy array (could also be a string or something else, in which case will not be converted to the npy format)
            # Generate file name
            if args.filename is None:
                npy_filename = args.path.split('/')
                npy_filename = npy_filename[len(npy_filename)-1].replace('.mat', '_') + args.section
            else:
                npy_filename = args.filename

            # Generate save path
            if args.savepath is None:
                args.savepath = ''
                s_path = args.path.split('/')
                s_path = s_path[:len(s_path)-1]
                for i in range(0, len(s_path)):
                    args.savepath += s_path[i] + '/'

            if args.savepath[-1] != '/':
                args.savepath += '/'

            if args.save:
                np.save(args.savepath + npy_filename, mat[args.section])  # Save mat file as npy file
                print('File saved to' + args.savepath + npy_filename + '.npy')
        else:
            print('The specified section does not contain a valid NumPy array. Exiting program.')
            sys.exit(-1)

    # Print (optional)
    if args.print:

        keys = [*mat.keys()]  # mat files contain many sections, or datasets; each key corresponds to a section

        print('There are', len(keys), 'sections in the mat file:', keys, '\n')

        for k in range(0, len(keys)):
            if isinstance(mat[keys[k]], (np.ndarray, np.generic)):
                print(str(k) + '. ' + keys[k].replace('__', '') + ' [shape ' + str(mat[keys[k]].shape) + ']:' + '\n' + str(mat[keys[k]]) + '\n')
            else:
                print(str(k) + '. ' + keys[k].replace('__', '') + ': ' + str(mat[keys[k]]) + '\n')

        if args.section is not None:
            np.set_printoptions(threshold=sys.maxsize)  # Print all section contents (otherwise numpy only prints a subset)
            print('\n', 'Chosen section content:', '\n', mat[args.section])

    return


if __name__ == '__main__':

    parser = ArgumentParser(description='mat to npy file converter. More info about the structure of mat file can be found at https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf')

    parser.add_argument('path', help='mat file path.')
    parser.add_argument('-sv', '--save', help='Save npy file (default: true).')
    parser.add_argument('-sp', '--savepath', help='npy file path.')
    parser.add_argument('-sc', '--section', help='Section to be converted.')
    parser.add_argument('-fn', '--filename', help='New file name.')
    parser.add_argument('-pr', '--print', help='Print mat file contents.')

    args = parser.parse_args()

    if args.save == 'False':
        args.save = False
    elif args.save == 'True':
        args.save = True

    if args.print == 'False':
        args.print = False
    elif args.print == 'True':
        args.print = True

    main(args)
