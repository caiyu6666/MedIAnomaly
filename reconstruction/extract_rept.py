from train import get_method
from options import Options


def main():
    opt = Options(isTrain=True)
    opt.parse()

    worker = get_method(opt)
    worker.set_gpu_device()
    worker.set_network_loss()
    worker.set_logging(test=True)
    worker.load_checkpoint()
    worker.set_dataloader()
    worker.data_rept()


if __name__ == "__main__":
    main()
