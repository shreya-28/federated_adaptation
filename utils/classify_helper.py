from utils.helper import Helper
import logging
import sys

from models.simple import FNN
from torch.utils.data import TensorDataset

logger = logging.getLogger("logger")


class ClassifyHelper(Helper):
    def create_model(self): #done
        local_model = FNN(name='Local',
                          created_time=self.params['current_time'])
        local_model.to(self.device)
        target_model = FNN(name='Target',
                           created_time=self.params['current_time'])
        target_model.to(self.device)

        if self.resumed_model:
            raise NotImplementedError("This hasnt been implemented. Train your model from the start")
        else:
            self.start_round = 1

        self.local_model = local_model
        self.target_model = target_model


    def load_data(self): #done for now
        logger.info('Loading data')

        self.train_data_x = torch.Tensor(path) # TODO
        self.train_data_y = torch.Tensor(path) # TODO

        self.test_data_x = torch.Tensor(path) # TODO
        self.test_data_y = torch.Tensor(path) # TODO

        self.train_dataset = TensorDataset(self.train_data_x, self.train_data_y)
        self.test_dataset = TensorDataset(self.test_data_x,self.test_data_y)

        if self.recreate_dataset:
            raise NotImplementedError("This hasnt been implemented. Cant do this recreation")
        else:
            pass

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

    def get_batch(self, train_data, bptt, evaluation=False): # done feels like there should be no change in this
        data, target = bptt
        data = data.to(self.device)
        target = target.to(self.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

