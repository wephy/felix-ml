import os
import math
import torch
import lightning.pytorch as pl
from models import BaseVAE
from models.types_ import *

class FelixExperiment(pl.LightningModule):
    def __init__(self,
                 model: BaseVAE,
                 params: dict) -> None:
        super(FelixExperiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        # self.hold_graph = False
        # try:
        #     self.hold_graph = self.params['retain_first_backpass']
        # except:
        #     pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, condition = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, condition=condition)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        return val_loss
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)

                # self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
#     def sample_images(self):
#         # Get sample reconstruction image            
#         test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
#         test_input = test_input.to(self.curr_device)
#         test_label = test_label.to(self.curr_device)

# #         test_input, test_label = batch
#         recons = self.model.generate(test_input, labels = test_label)
#         vutils.save_image(recons.data,
#                           os.path.join(self.logger.log_dir, 
#                                        "Reconstructions", 
#                                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
#                           normalize=True,
#                           nrow=8)

#         try:
#             samples = self.model.sample(64,
#                                         self.curr_device,
#                                         labels = test_label)
#             vutils.save_image(samples.cpu().data,
#                               os.path.join(self.logger.log_dir, 
#                                            "Samples",      
#                                            f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
#                               normalize=True,
#                               nrow=8)
#         except Warning:
#             pass
