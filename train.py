import argparse
import models

parser = argparse.ArgumentParser()

# data directory
parser.add_argument('data_dir', action="store")

# architecture
parser.add_argument('--arch', action="store", help= 'the network architecture: VGG16 or resnet18', default='vgg16',  dest='arch')

# hyperparameters
parser.add_argument('--hidden_units', action="store", help= 'number of hidden units in activation layer', default=1024,  dest='hidden_units')
parser.add_argument('--learning_rate', action="store", help= 'learning rate', default=0.001,  dest='learning_rate')
parser.add_argument('--epochs', action="store", help= 'number of training epochs', default=20,  dest='epochs')
parser.add_argument('--batch_size', action="store", help= 'number of training epochs', default=64,  dest='batch_size')
parser.add_argument('--save_dir', action="store", help= 'save directory', default='./',  dest='save_dir')
parser.add_argument('--optim', action="store", help= 'optimizer, SGD or Adam', default='SGD',  dest='optim')

# using gpu
parser.add_argument('--gpu', action="store_true", help= 'add to train on gpu', default=False,  dest='train_on_gpu')

# retrieve the results
results = parser.parse_args()

data_dir = results.data_dir
learning_rate = results.learning_rate
epochs = results.epochs
train_on_gpu = results.train_on_gpu
hidden_units = results.hidden_units
arch = results.arch
batch_size = results.batch_size
save_dir = results.save_dir
optim = results.optim

models.train(data_dir, lr = learning_rate, train_on_gpu = train_on_gpu, epochs = epochs, hidden_units = hidden_units,
             arch = arch, batch_size = batch_size, save_dir = save_dir, optim = optim)
