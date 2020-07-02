from utils.helper import Helper
import logging
import sys
import numpy as np
import torch
from collections import defaultdict
from models.simple import FNN
from torch.utils.data import TensorDataset
import random
logger = logging.getLogger("logger")


class ClassifyHelper(Helper):
    def create_model(self): #done
        local_model = FNN(input_size = self.no_features, hidden_size=100,num_classes=2, name='Local',
                          created_time=self.params['current_time'])
        local_model.to(self.device)
        target_model = FNN(input_size = self.no_features, hidden_size=100,num_classes=2, name='Target',
                           created_time=self.params['current_time'])
        target_model.to(self.device)

        if self.resumed_model:
            raise NotImplementedError("This hasnt been implemented. Train your model from the start")
        else:
            self.start_round = 1

        self.local_model = local_model
        self.target_model = target_model


    def load_data(self): #done for now. Need to check if the stamements in else have some effect on the rest of the code
        logger.info('Loading data')

        numpy_train_x = np.load(f"{self.params['repo_path']}/data/train_x.npy")
        numpy_train_y = np.load(f"{self.params['repo_path']}/data/train_y.npy")
        numpy_test_x = np.load(f"{self.params['repo_path']}/data/test_x.npy")
        numpy_test_y = np.load(f"{self.params['repo_path']}/data/test_y.npy")
        temp=[]
        for i in range(len(numpy_train_y)):
            temp.append(np.argmax(numpy_train_y[i]))
        numpy_train_y = np.array(temp)
        temp=[]
        for i in range(len(numpy_test_y)):
            temp.append(np.argmax(numpy_test_y[i]))
        numpy_test_y = np.array(temp)

        self.no_features = numpy_train_x.shape[1]
        train_data_x = torch.Tensor(numpy_train_x) 
        train_data_y = torch.Tensor(numpy_train_y) 

        test_data_x = torch.Tensor(numpy_test_x) 
        test_data_y = torch.Tensor(numpy_test_y) 
        

        self.train_dataset = TensorDataset(train_data_x, train_data_y)
        self.test_dataset = TensorDataset(test_data_x,test_data_y)

        self.train_dataset = TensorDataset(train_data_x, train_data_y)
        self.test_dataset = TensorDataset(test_data_x,test_data_y)
        train_data_path = f"{self.params['repo_path']}/data/train_data.pt.tar"     
        train_image_weight_path = f"{self.params['repo_path']}/data/train_image_weight.pt"
        auxiliary_data_path = f"{self.params['repo_path']}/data/auxiliary_data.pt.tar"
        test_data_path = f"{self.params['repo_path']}/data/test_data.pt.tar"

        if self.recreate_dataset:
            indices_per_participant, train_image_weight = self.sample_dirichlet_data(self.train_dataset,
                self.params['number_of_total_participants'],
                alpha=0.9)
            self.train_data = [(user, self.get_train(indices_per_participant[user])) for user in range(self.params['number_of_total_participants'])]
            self.train_image_weight = train_image_weight
            auxiliary_index_intest = random.sample(list(range(len(self.test_dataset))), len(self.test_dataset)//10)
            test_index_remove_auxiliary = [elem for elem in range(len(self.test_dataset)) if elem not in auxiliary_index_intest] 
            self.auxiliary_data = self.get_test(auxiliary_index_intest)
            self.test_data = self.get_test(test_index_remove_auxiliary)
            torch.save(self.train_data, train_data_path)
            torch.save(self.train_image_weight, train_image_weight_path)
            torch.save(self.auxiliary_data, auxiliary_data_path)
            torch.save(self.test_data, test_data_path)
        
        else:
            self.train_data = torch.load(train_data_path)
            self.train_image_weight = torch.load(train_image_weight_path)
            self.test_data = torch.load(test_data_path)

    def get_test(self, indices):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        return test_loader

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices))
        return train_loader

    def get_batch(self, train_data, bptt, evaluation=False): # done. feels like there should be no change in this
        print(bptt)
        data, target = bptt
        data = data.to(self.device)
        target=target.long()
        target = target.to(self.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)

        return data, target

    def sample_dirichlet_data(self, dataset, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(dataset):
            _, label = x
            label = int(label.numpy())
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        print(cifar_classes.keys())
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())
        class_size = len(cifar_classes[0])
        datasize = {}
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                datasize[user, n] = no_imgs
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        train_img_size = np.zeros(no_participants)
        print(datasize)
        for i in range(no_participants):
            train_img_size[i] = sum([datasize[i,j] for j in range(2)])
        clas_weight = np.zeros((no_participants,2))
        for i in range(no_participants):
            for j in range(2):
                clas_weight[i,j] = float(datasize[i,j])/float(train_img_size[i])
        return per_participant_list, clas_weight
