from utils.helper import Helper
import logging

from models.simple import FNN

logger = logging.getLogger("logger")

class ClassifyHelper(Helper):
    def create_model(self):
        local_model = FNN(name='Local',
                          created_time=self.params['current_time'])
        local_model.to(self.device)
        target_model = FNN(name='Target',
                           created_time=self.params['current_time'])
        target_model.to(self.device)

        ##TODO

        self.local_model = local_model
        self.target_model = target_model


    def load_data(self):
        logger.info('Loading data')

        #data load TODO

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

    def get_batch():
        #TODO
