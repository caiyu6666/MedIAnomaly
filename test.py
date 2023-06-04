from utils.ae_worker import AEWorker
from options import Options


def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    worker = AEWorker(opt)
    worker.set_gpu_device()
    worker.set_seed()
    # worker.set_network()
    worker.set_network_loss()
    worker.load_checkpoint()
    worker.set_test_loader()
    worker.run_eval()


if __name__ == "__main__":
    main()
