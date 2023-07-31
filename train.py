from utils.ae_worker import AEWorker
from utils.memae_worker import MemAEWorker
from utils.aeu_worker import AEUWorker
from utils.vae_worker import VAEWorker
from utils.ganomaly_worker import GanomalyWorker
from options import Options


def get_method(opt):
    print("=> Running Method: {}".format(opt.model['name']))
    if opt.model['name'] == 'ae' or opt.model['name'] == 'ae-grad':
        return AEWorker(opt)
    elif opt.model['name'] == 'memae':
        return MemAEWorker(opt)
    elif opt.model['name'] == 'aeu':
        return AEUWorker(opt)
    # elif opt.model['name'] == 'vae':
    elif 'vae' in opt.model['name']:
        return VAEWorker(opt)
    elif opt.model['name'] == 'ganomaly':
        return GanomalyWorker(opt)
    else:
        raise Exception("Invalid model name: {}".format(opt.model['name']))


def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    worker = get_method(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_network_loss()
    # worker.set_network()
    # worker.set_loss()
    worker.set_optimizer()
    worker.set_dataloader()
    worker.set_logging()
    worker.run_train()


if __name__ == "__main__":
    main()
