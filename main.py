import models.mobinet as mobinet
import models.resnet18 as resnet18
import models.resnet50 as resnet50
import util.prepare_data as util


def main():
    print("[*] starting...")
    epochs = 50
    lr = 0.01
    momentum = 0.5

    device = util.prepare_cuda()

    train_loader, test_loader = util.get_tiny_imagenet_dataloaders()


    resnet18.proceed_ptq(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False)
    resnet18.proceed_qat(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False)

    resnet50.proceed_ptq(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False)
    resnet50.proceed_qat(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False)

    mobinet.proceed_ptq(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False)
    mobinet.proceed_qat(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False)

    print("[+] finished all tasks, hell yeah :sunglasses:")


if __name__ == '__main__':
    main()
