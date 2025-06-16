import models.mobinet as mobinet
import models.resnet18 as resnet18
import models.resnet50 as resnet50
import util.prepare_data as util


def main():
    print("[*] starting...")
    epochs = 1
    lr = 0.01
    momentum = 0.5

    device = util.prepare_cuda()

    train_loader, test_loader = util.get_tiny_imagenet_dataloaders()

    resnet18.proceed(device, epochs, train_loader, test_loader, lr, momentum)
    # resnet50.proceed(device, epochs, train_loader, test_loader, lr, momentum)
    # mobinet.proceed(device, epochs, train_loader, test_loader, lr, momentum)

    print("[+] finished all tasks, hell yeah :sunglasses:")


if __name__ == '__main__':
    main()
