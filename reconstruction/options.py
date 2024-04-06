import os
import argparse


class Options:
    def __init__(self, isTrain):
        self.project_name = None
        self.dataset = None
        self.fold = None
        self.result_dir = None
        self.isTrain = isTrain
        self.model = dict()
        self.train = dict()
        self.test = dict()
        self.transform = dict()
        self.post = dict()
        self.gpu = None
        # self.tags = None
        # self.notes = None

        self.data_name = {'rsna': 'RSNA', 'vin': 'VinDr-CXR', 'brain': 'Brain Tumor', 'lag': 'LAG', 'isic': 'ISIC2018',
                          'c16': 'Camelyon16', 'brats': 'BraTS2021'}
        self.epochs = {'rsna': 250, 'vin': 250, 'brain': 600, 'lag': 250, 'brats': 250,
                       'oct': 250, 'colon': 250}
        self.in_c = {'c16': 3, 'colon': 1}

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument("-d", '--dataset', type=str, default='rsna',
                            help='rsna, vin, brain, lag, brats, oct, colon, isic')
        parser.add_argument("-g", '--gpu', type=int, default=6, help='select gpu devices')
        parser.add_argument("-p", '--project-name', type=str, default="MedIAnomaly", required=False,
                            help='Name of the current project. eg, MedIAnomaly')
        parser.add_argument("-n", '--notes', type=str, default="default", required=False,
                            help='Notes of the current experiment. e.g., ae-architecture')
        parser.add_argument("-f", '--fold', type=str, default='0', help='0-4, five fold cross validation')
        parser.add_argument("-m", '--model-name', type=str, default='ae', help='ae, aeu, memae')
        parser.add_argument('--input-size', type=int, default=64, help='input size of the image')

        # Parameters only for reconstruction model
        parser.add_argument('--base-width', type=int, default=16,
                            help='Base channels of CNN layers. Please do not modify this value, and adjust the '
                                 'expansion instead.')
        parser.add_argument('--expansion', type=int, default=1, help='expansion of the base channels.')
        parser.add_argument('--hidden-num', type=int, default=1024, help='Hidden size of the bottleneck')
        parser.add_argument("-ls", '--latent-size', type=int, default=16,
                            help='latent size of the reconstruction model')
        parser.add_argument('--en-depth', type=int, default=1, help='Depth of each encoder block')
        parser.add_argument('--de-depth', type=int, default=1, help='Depth of each decoder block')

        parser.add_argument('--train-epochs', type=int, default=250, help='number of training epochs')
        parser.add_argument('--train-eval-freq', type=int, default=25, help='epoch to evaluate')
        parser.add_argument('-bs', '--train-batch-size', type=int, default=64, help='batch size')
        parser.add_argument('--train-lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--train-weight-decay', type=float, default=0, help='weight decay')
        parser.add_argument('--train-seed', type=int, default=None, help='random seed')

        parser.add_argument("-save", '--test-save-flag', action='store_true')
        parser.add_argument('--test-model-path', type=str, default=None, help='model path to test')

        args = parser.parse_args()

        self.gpu = args.gpu
        self.dataset = args.dataset
        self.project_name = args.project_name
        self.fold = args.fold
        self.result_dir = os.path.expanduser("~") + f'/Experiment/MedIAnomaly/{self.dataset}'

        self.model['name'] = args.model_name
        self.model['in_c'] = self.in_c.setdefault(self.dataset, 1)
        self.model['input_size'] = args.input_size

        # Parameters only for reconstruction model
        self.model['base_width'] = args.base_width
        self.model['expansion'] = args.expansion
        self.model['hidden_num'] = args.hidden_num
        self.model['ls'] = args.latent_size
        self.model['en_depth'] = args.en_depth
        self.model['de_depth'] = args.de_depth

        # --- training params --- #
        self.train['save_dir'] = '{}/{}/fold_{}'.format(self.result_dir, self.model['name'], self.fold)
        self.train['epochs'] = self.epochs.setdefault(self.dataset, 250)
        self.train['eval_freq'] = args.train_eval_freq
        self.train['batch_size'] = args.train_batch_size
        self.train['lr'] = args.train_lr
        self.train['weight_decay'] = args.train_weight_decay
        self.train['seed'] = args.train_seed

        # --- test parameters --- #
        self.test['save_flag'] = args.test_save_flag
        self.test['save_dir'] = '{:s}/test_results'.format(self.train['save_dir'])
        if not args.test_model_path:
            self.test['model_path'] = '{:s}/checkpoints/model.pth'.format(self.train['save_dir'])

    def save_options(self):
        if not os.path.exists(self.train['save_dir']):
            os.makedirs(self.train['save_dir'], exist_ok=True)
            os.makedirs(os.path.join(self.train['save_dir'], 'test_results'), exist_ok=True)
            os.makedirs(os.path.join(self.train['save_dir'], 'checkpoints'), exist_ok=True)

        filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        file = open(filename, 'w')
        groups = ['model', 'train', 'test', 'transform']

        file.write("# ---------- Options ---------- #")
        file.write('\ndataset: {:s}\n'.format(self.dataset))
        file.write('isTrain: {}\n'.format(self.isTrain))
        for group, options in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in options.items():
                    if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                        file.write("{:s}:\n".format(name))
                        for t_val in val.transforms:
                            file.write("\t{:s}\n".format(t_val.__class__.__name__))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()
