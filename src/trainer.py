import config

from tqdm import tqdm
import torch
import random

class Trainer:
    def __init__(self,
                model,
                optimizer,
                dataloader_train,
                scheduler,
                scaler,
                rank,
                writer):
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.writer = writer
        self.scheduler = scheduler
        self.scaler = scaler
        self.rank = rank

        self.epoch = 0
    
    def train(self):
        self.model.train()

        num_batches = 0
        loss_epoch = 0.0

        self.optimizer.zero_grad()

        for data in tqdm(self.dataloader_train, disable=self.rank != 0):

            loss, _ = self.load_data_use_model_compute_loss(data, validation=False)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            num_batches += 1
            loss_epoch += float(loss.item())

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
        self.log_losses_tensorboard(loss_epoch, num_batches, 'train_')

    def train_for_epochs(self, epochs):
        for epoch in range(epochs):
            if self.rank == 0:
                print(f"EPOCH {epoch}")
            self.epoch = epoch
            self.train()
            torch.cuda.empty_cache()

            self.scheduler.step()
            if self.rank == 0:
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)


    def load_data_use_model_compute_loss(self, data, validation=False):
        # Elements inside the data dict are already tensors! Magic!
            
        """
        for data_folder in config.DATASET_FOLDER_STRUCT:
            data_name, data_extension = data_folder
            data[data_name].to(self.device)
        """
        source, targets = data
        source = source.to(self.rank)
        targets = targets.to(self.rank)

        prediction = self.model(source)

        loss = self.model.module.compute_loss(prediction, targets)
        difference = prediction-targets
        if self.rank == 0:
            self.writer.add_scalar('difference', difference[0].cpu(), self.epoch)
        # end TMP

        return loss, None


    def log_losses_tensorboard(self, loss_epoch, num_batches, prefix=''):
        if self.rank == 0:
            self.writer.add_scalar(prefix + 'loss_total', loss_epoch / num_batches, self.epoch)
            
            """
            for key, value in detailed_losses_epoch.items():
                self.writer.add_scalar(prefix + key, value / num_batches, self.epoch)
            """