from tqdm import tqdm
import torch

class Trainer:
    def __init__(self,
                model,
                optimizer,
                dataloader_train,
                dataloader_val,
                writer,
                device,
                scheduler,
                scaler,
                rank,
                world_size,
                cur_epoch):
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.writer = writer
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler
        self.rank = rank
        self.world_size = world_size
        self.cur_epoch = cur_epoch

        self.detailed_loss_weights = {
            'loss_wp': 1.0,
            'loss_target_speed': 1.0,
            'loss_checkpoint': 1.0,
            'loss_semantic': 1.0,
            'loss_bev_semantic': 1.0,
            'loss_depth': 1.0,
            'loss_center_heatmap': 1.0,
            'loss_wh': 1.0,
            'loss_offset': 1.0,
            'loss_yaw_class': 1.0,
            'loss_yaw_res': 1.0,
            'loss_velocity': 1.0,
            'loss_brake': 1.0,
            'loss_forcast': 0.2,
            'loss_selection': 0.0,
        }
    
    def train(self):
        self.model.train()

        num_batches = 0
        loss_epoch = 0.0

        self.optimizer.zero_grad(set_to_none=False)
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_loss_weights}

        for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                losses, _ = self.load_data_compute_loss(data, validation=False)
                loss = torch.zeros(1, dtype=torch.float32, device=self.device)

                for key, value in losses.items():
                    loss += self.detailed_loss_weights[key] * value
                    detailed_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            num_batches += 1
            loss_epoch += float(loss.item())

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
        self.log_losses_tensorboard(loss_epoch, detailed_losses_epoch, num_batches, 'train_')

    def load_data_compute_loss(data, validation=False):
        pass

    def log_losses_tensorboard(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        if self.rank == 0:
            self.writer.add_scalar(prefix + 'loss_total', loss_epoch / num_batches, self.cur_epoch)

            for key, value in detailed_losses_epoch.items():
                self.writer.add_scalar(prefix + key, value / num_batches, self.cur_epoch)
