from train import get_method
from options import Options


def main():
    opt = Options(isTrain=True)
    opt.parse()
    # opt.save_options()

    worker = get_method(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_network_loss()
    worker.set_logging(test=True)
    worker.load_checkpoint()
    worker.set_dataloader(test=True)
    worker.run_eval()

    # worker.data_rept()


if __name__ == "__main__":
    main()
