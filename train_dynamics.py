import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model import MLPdynamics, Transformerdynamics
from src.dataset_loader import F1TENTH_Dataset, F1TENTH_DataLoader

import hydra
from src.logger import Logger
import src.utils as utils
from tqdm import tqdm


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # setup
        group_name = self.cfg.exp_prefix + '_' + self.cfg.model.name + '_' + self.cfg.exp_suffix
        self.logger = Logger(project_name='f1tenth',
                            log_dir=self.work_dir, 
                            cfg=self.cfg,
                             use_tb=self.cfg.use_tb, 
                             use_wandb=self.cfg.use_wandb, 
                             group_name=group_name)

        # dataset
        self.train_dataset = F1TENTH_Dataset(self.cfg.rootdir, 
                                             self.cfg.train_data_path,
                                             self.cfg.model.state_dim,
                                            self.cfg.model.action_dim,
                                             history_len=self.cfg.model.history_len)
        self.eval_dataset = F1TENTH_Dataset(self.cfg.rootdir, 
                                            self.cfg.test_data_path,
                                             self.cfg.model.state_dim,
                                            self.cfg.model.action_dim,
                                            history_len=self.cfg.model.history_len)
        self.train_loader = F1TENTH_DataLoader(self.train_dataset, 
                                               batch_size=self.cfg.model.batch_size, 
                                               shuffle=True, 
                                               num_workers=self.cfg.num_workers)
        self.eval_loader = F1TENTH_DataLoader(self.eval_dataset,
                                              batch_size=self.cfg.model.batch_size,
                                              shuffle=False,
                                              num_workers=self.cfg.num_workers)

        print(f'Train dataset: {len(self.train_dataset)}')
        print(f'Eval dataset: {len(self.eval_dataset)}')

        # model
        if self.cfg.model.name == 'mlp':
            self.model = MLPdynamics(
                                        state_dim=self.cfg.model.state_dim,
                                        act_dim=self.cfg.model.action_dim,
                                        out_dim=self.cfg.model.state_out_dim,
                                        history_len=self.cfg.model.history_len,
                                        hidden_dim=self.cfg.model.hidden_dim
                                     ).to(self.device)
        elif self.cfg.model.name == 'Transformer':
            self.model = Transformerdynamics(
                                                state_dim=self.cfg.model.state_dim,
                                                act_dim=self.cfg.model.action_dim,
                                                history_len=self.cfg.model.history_len,
                                                hidden_dim=self.cfg.model.hidden_dim,
                                                nhead=self.cfg.model.nheads,
                                                num_layers=self.cfg.model.num_layers
                                             ).to(self.device)
        else:
            raise ValueError(f'Invalid model name: {self.cfg.model.name}')

        print(f'Model: {self.model} has {self.model.parameter_num()} parameters.')

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)
        ## scheduler
        if self.cfg.model.lr_scheduler == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                       step_size=self.cfg.model.lr_step_size, 
                                                       gamma=self.cfg.model.lr_gamma)

    def train_step(self, train_batch):
        state, action, next_state = train_batch
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)

        pred_next_state = self.model(state, action)
        loss = nn.MSELoss()(pred_next_state, next_state)

        if self.cfg.model.pinn:
            physics_next_state = utils.single_track_dynamics(state[:, -1, :], action)
            physics_loss = nn.MSELoss()(physics_next_state[..., :4], next_state[..., :4])
            loss += self.cfg.model.pinn_lambda * physics_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self._global_step > 30 and loss.item() > 5.0:
            #import pdb; pdb.set_trace()
            print('Loss is too high, loss:', loss.item())
            loss_each = ((next_state - pred_next_state)**2).sum(axis=1)
            idx = np.argmax(loss_each.cpu().detach().numpy())
            print('State:', state[idx, -1])
            print('Action:', action[idx, -1])
            print('Next State:', next_state[idx])
            print('Predicted Next State:', pred_next_state[idx])


        if self._global_step % self.cfg.log_interval == 0:
            self.logger.log('train/loss', loss.item(), self._global_step)
        self._global_step += 1

        return loss.item()

    def train(self):
        for epoch in tqdm(range(self.cfg.model.num_epochs)):
            for train_batch in self.train_loader:
                self.model.train()
                loss = self.train_step(train_batch)

            print(f'Epoch: {epoch}, Loss: {loss}')

            if epoch % self.cfg.eval_interval == 0:
                self.evaluate()

            if hasattr(self, 'scheduler') and self.cfg.model.lr_scheduler == 'StepLR':
                self.scheduler.step()
                self.logger.log('train/lr', self.scheduler.get_last_lr()[0], self._global_step)

            if epoch % self.cfg.save_checkpoint_interval == 0:
                self.save_model()

    def save_model(self):
        save_path = self.work_dir / 'checkpoints'
        save_path.mkdir(exist_ok=True, parents=True)
        save_path = save_path / f'{self.cfg.model.name}_model_{self._global_step}.pth'
        torch.save(self.model, save_path)
        print(f'Model saved at {save_path}')

    def evaluate(self):
        for eval_batch in self.eval_loader:
            self.model.eval()
            state, action, next_state = eval_batch
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)

            with torch.no_grad():
                pred_next_state = self.model(state, action)
                loss = nn.MSELoss()(pred_next_state, next_state)
                self.logger.log('eval/loss', loss.item(), self._global_step)


@hydra.main(config_path='cfgs', config_name='config', version_base='1.1')
def main(cfg):
    #from train_dynamics import Workspace as W
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()