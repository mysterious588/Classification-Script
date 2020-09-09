# Imports here
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os

def train(data_dir, lr = 0.001, arch = 'vgg16', hidden_units = 1024, optim = 'SGD', batch_size = 64,
          loss= 'cross_entropy', train_on_gpu= False, epochs = 10, save_dir = './'):
        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    num_classes = len(os.listdir(train_dir))

    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Crop and Resize the data and validation images in order to be able to be fed into the network
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    train_datasets = ImageFolder( train_dir, transform = train_transforms)
    test_datasets = ImageFolder(test_dir, transform = test_transforms)
    valid_datasets = ImageFolder(valid_dir, transform = test_transforms)

    trainloader = DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
    testloader = DataLoader(test_datasets, batch_size = batch_size, shuffle = True)
    validloader = DataLoader(valid_datasets, batch_size = batch_size, shuffle = True)

    if arch == 'vgg16':
        from torchvision.models import vgg16   
        model = vgg16(pretrained = True)
        for param in model.parameters():
            param.require_grad = False
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hidden_units, num_classes))
        params = model.classifier.parameters()
        
    elif arch == 'resnet18':
        from torchvision.models import resnet18          
        model = resnet18(pretrained = True)
        for param in model.parameters():
            param.require_grad = False   
        params = model.parameters()
        model.fc = nn.Sequential(nn.Linear(512, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hidden_units, num_classes))

        params = model.fc.parameters()

    else:
        raise Exception('architecture not available') 
            
    if loss == 'cross_entropy':    
            criterion = nn.CrossEntropyLoss()
    else:
        raise Exception('loss function not available') 
        
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr = lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr = lr)
    else:
        raise Exception('optimizer must be SGD or Adam') 
    
    if (train_on_gpu):
        model.cuda()
        
    valid_loss_min = np.inf
    try:
        print('************************** TRAINING STARTED ***************************')
        print('******* use ctrl + c to abort and save the best validated model *******')        
        for epoch in range(epochs + 1):
            train_loss = 0
            valid_loss = 0

            model.train()
            for data, target in trainloader:
                # move to gpu if available
                if(train_on_gpu):
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)
                
            # validation
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))            
            model.eval()
            for data, target in validloader:
                # move to gpu if available
                if(train_on_gpu):
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)

                _, predection = torch.max(output, 1)    
                correct_tensor = predection.eq(target.data.view_as(predection))
                correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

                # calculate test accuracy for each object class
                for i in range(len(target.data)):
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

            # calculate average losses
            train_loss = train_loss/len(trainloader.sampler)
            valid_loss = valid_loss/len(validloader.sampler)

            acc = 100. * np.sum(class_correct) / np.sum(class_total)

            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tAccuracy: {:0.2f}%'.format(
                epoch, train_loss, valid_loss, acc))
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save({'state_dict':model.state_dict(),
                           'arch': arch,
                           'hidden_units': hidden_units,
                           'num_classes': num_classes},
                           save_dir+'/trained_model.pt')
                valid_loss_min = valid_loss
    except KeyboardInterrupt:
        print('training stopped, the last model has been saved')
        

def load_model(path, map_location):
    checkpoint = torch.load(path, map_location)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    num_classes = checkpoint['num_classes']
    
    if arch == 'vgg16':
        from torchvision.models import vgg16
        model = vgg16(pretrained = True)
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_units, num_classes))

        model.load_state_dict(checkpoint['state_dict'])
        return model
    elif arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained = True)
        model.fc = nn.Sequential(nn.Linear(512, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_units, num_classes))

        model.load_state_dict(checkpoint['state_dict'])
        return model        

def predict(image_path, model, topk=5, train_on_gpu = False, category_names = None):
    
    if train_on_gpu:
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    model = load_model(model, map_location)
    
    from PIL import Image
    image = Image.open(image_path)
    
    transform = transforms.Compose([transforms.CenterCrop(256),
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(std=[0.229, 0.224, 0.225],
                                                            mean=[0.485, 0.456, 0.406] )])
    
    image = transform(image)        
    image = torch.stack((image,))

    if train_on_gpu:
        image = image.cuda()
        model = model.cuda()
        
    model.eval()
    output = model(image)
    
    probs, classes = output.topk(topk)[0].squeeze().tolist(), output.topk(topk)[1].squeeze().tolist()
    
    #print('top ' + str(topk) + ' classes: ', classes)
    
    from prettytable import PrettyTable
    
    t = PrettyTable(['class', 'prob'])
    for i in range(topk):
        t.add_row([classes[i], probs[i]])
    print(t)
        
    if (category_names != None):
        import json

        with open(category_names, 'r') as f:
            category_names = json.load(f)
        
        classes = [category_names[str(i+1)] for i in classes]
        
        t = PrettyTable(['class', 'prob'])
        for i in range(topk):
            t.add_row([classes[i], probs[i]]) 
        print(t)         