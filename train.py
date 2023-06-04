from utils.ae_worker import AEWorker
from utils.memae_worker import MemAEWorker
from utils.aeu_worker import AEUWorker
from options import Options


def get_method(opt):
    print("=> Running Method: {}".format(opt.model['name']))
    if opt.model['name'] == 'ae':
        return AEWorker(opt)
    elif opt.model['name'] == 'memae':
        return MemAEWorker(opt)
    elif opt.model['name'] == 'aeu':
        return AEUWorker(opt)
    else:
        raise Exception("Invalid model name: {}".format(opt.model['name']))


def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    # worker = AEWorker(opt)
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
