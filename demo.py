import argparse
from utils_data import *
from utils_algo import *
from models import *
import warnings

# from torchvision import datasets, transformsz

# datasets.MNIST.resources = [
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c'),
# ]

# transform = transforms.ToTensor()
# train_ds = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
# test_ds  = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)

warnings.filterwarnings("ignore")

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='Demo file for Sconf',
	usage='Demo with similarity-confidence.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=3000, type=int)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=60)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-3, type=float)
parser.add_argument('-me', '--method', help='method name', choices=['u', 'nn', 'abs', 'sd', 'siamese', 'contrastive'], type=str, required=True, default = 'u')


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


if args.method=='u' or args.method=='abs'or args.method=='nn':
    v_train_loader, train_loader, test_loader, sd_train_loader, pair_train_loader, prior = prepare_mnist_data(batch_size=args.batch_size)
    model = mlp_model(input_dim=28*28,hidden_dim=500,output_dim=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr = args.learning_rate)
    test_accuracy = accuracy_check(loader=test_loader, model=model).to("cpu")
    print('Epoch: 0. Teet Accuracy: {}%'.format(test_accuracy.numpy()[0]))
    epoch=0
    for epoch in range(1,args.epochs):
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']/10
        if epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']/10
        for i, (images, sconf) in enumerate(train_loader):
            images, sconf = images.to(device), sconf.to(device)
            optimizer.zero_grad()
            outputs = model(images).to(device)
            loss = Sconf_loss(f=outputs, prior=prior, sconf=sconf, loss_name = args.method)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_accuracy = accuracy_check(loader=test_loader, model=model).to("cpu")
            print('Epoch: {}. Test Accuracy: {}%'.format(epoch, test_accuracy.numpy()[0]))

if args.method=='sd':
    v_train_loader, train_loader,test_loader,sd_train_loader, pair_train_loader, prior = prepare_mnist_data(batch_size=args.batch_size)
    model = mlp_model(input_dim=28*28,output_dim=1,hidden_dim=500)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr = args.learning_rate)
    test_accuracy = accuracy_check(loader=test_loader, model=model).to("cpu")
    print('Epoch: 0. Test Accuracy: {}%'.format(test_accuracy.numpy()[0]))
    epoch=0
    for epoch in range(1,args.epochs):
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']/10
        if epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']/10
        for i, (images, sd_label) in enumerate(sd_train_loader):
            images, sd_label = images.to(device), sd_label.to(device)
            optimizer.zero_grad()
            outputs = model(images).to(device)
            loss = SD_loss(f=outputs, prior=prior, sd_label=sd_label)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_accuracy = accuracy_check(loader=test_loader, model=model).to("cpu")
            print('Epoch: {}. Test Accuracy: {}%'.format(epoch, test_accuracy.numpy()[0]))

if args.method == 'siamese':
    v_train_loader, train_loader, test_loader, sd_train_loader, pair_train_loader, prior = prepare_mnist_data(batch_size=args.batch_size)
    model = Siamese_net(mlp_model(input_dim=28*28, hidden_dim=500, output_dim=300))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    for epoch in range(1, 10):
        for i, (images,images2, sd) in enumerate(pair_train_loader):
            images, images2, sd = images.to(device),images2.to(device) , sd.to(device)
            optimizer.zero_grad()
            outputs, outputs2 = model(images,images2)
            outputs = outputs.to(device)
            outputs2 = outputs2.to(device)
            loss = siamese_loss(f1=outputs, f2=outputs2, sd_label=sd)
            loss.backward()
            optimizer.step()
    image=[]
    for i, (image, label) in enumerate(v_train_loader):
        continue
    image=image.to(device)
    label=label.to(device)
    rep_tr = model.sub_forward(image).data.cpu().numpy()
    f=model.sub_forward(image).to(device)
    fp=[]
    fn=[]
    for i in range(len(label)):
        if label[i] == 1:
            fp=f[i]
        else:
            fn=f[i]
    rep_t = model.sub_forward(image).data.cpu().numpy()
    label_t = label.cpu().numpy()

    print('Test Accuracy: {}%', format(check_sia(test_loader, fp, fn, model) * 100))


if args.method == 'contrastive':
    v_train_loader, train_loader, test_loader, sd_train_loader, pair_train_loader, prior = prepare_mnist_data(batch_size=args.batch_size)
    model = Siamese_net(mlp_model(input_dim=28*28, hidden_dim=500, output_dim=300))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    for epoch in range(1, 10):
        for i, (images, images2, sd) in enumerate(pair_train_loader):
            images, images2, sd = images.to(device), images2.to(device), sd.to(device)
            optimizer.zero_grad()
            outputs, outputs2 = model(images, images2)
            outputs = outputs.to(device)
            outputs2 = outputs2.to(device)
            loss = contrastive_loss(f1=outputs, f2=outputs2, sd_label=sd)
            loss.backward()
            optimizer.step()
    image = []
    for i, (image, label) in enumerate(v_train_loader):
        continue
    image = image.to(device)
    label = label.to(device)
    rep_tr = model.sub_forward(image).data.cpu().numpy()
    f = model.sub_forward(image).to(device)
    fp = []
    fn = []
    for i in range(len(label)):
        if label[i] == 1:
            fp = f[i]
        else:
            fn = f[i]
    rep_t = model.sub_forward(image).data.cpu().numpy()
    label_t = label.cpu().numpy()
    print('Test Accuracy: {}%'.format(check_sia(test_loader, fp, fn, model)*100))
