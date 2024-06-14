import shutil
import random
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn as nn

class TensorboardWriter:
    def __init__(self, log_dir):
        self.train_writer = SummaryWriter(os.path.join(log_dir, "train"))
        self.valid_writer = SummaryWriter(os.path.join(log_dir, "valid"))

    def train_log_step(self, name, value, step, same=False):
        if same:
            self.train_writer.add_scalar(f"{name}/step", value, step)
        self.train_writer.add_scalar(f"train/{name}/step", value, step)

    def train_log_epoch(self, name, value, epoch, same=True):
        if same:
            self.train_writer.add_scalar(f"{name}/epoch", value, epoch)
        self.train_writer.add_scalar(f"train/{name}/epoch", value, epoch)

    def valid_log_step(self, name, value, step, same=False):
        if same:
            self.valid_writer.add_scalar(f"{name}/step", value, step)
        self.valid_writer.add_scalar(f"valid/{name}/step", value, step)

    def valid_log_epoch(self, name, value, epoch, same=True):
        if same:
            self.valid_writer.add_scalar(f"{name}/epoch", value, epoch)
        self.valid_writer.add_scalar(f"valid/{name}/epoch", value, epoch)
        

class Trainer:

    def __init__(
        self,
        model,
        train_data_loader,
        valid_data_loader,
        config=None,
        log_dir="logs/"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ):
        print("Setup training")
        self.max_updates = config["max_updates"]
        self.max_epochs = config["max_epochs"]
        self.log_interval_updates = config["log_interval_updates"]
        self.save_interval_updates = config["save_interval_updates"]
        self.keep_interval_updates = config["keep_interval_updates"]
        self.device = config["device"]
        self.log_dir = log_dir
        self.tensorboard = TensorboardWriter(log_dir=log_dir)
        self.model = model.to(self.device)
        self.valid_data_loader = valid_data_loader
        self.train_data_loader = train_data_loader

        self.best_loss = None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            **config["optimizer"]
        )
        self.lr = config["optimizer"]["lr"]
        self.lr_scheduler = None

        if config["random_seed"] is not None:
            random.seed(config["random_seed"])
            torch.manual_seed(config["random_seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(config["random_seed"])

        self.current_epoch = 0
        self.current_step = 0

        self.last_saved_checkpoints = []

    def get_model_state(self):
        return {
            "epoch": self.current_epoch,
            "lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else self.lr,
            "optimizer": self.optimizer.state_dict(),
            "state_dict": self.model.state_dict(),
            "step": self.current_step
        }

    def save_checkpoint(self, is_last=True, is_best=False, step=None):
        print(f"Saving checkpoint at {self.log_dir} "
                    f"[is_last={is_last}, is_best={is_best}, step={step}]")
        if is_last:
            last_fpath = os.path.join(self.log_dir, "checkpoint_last.pt")
            torch.save(self.get_model_state(), last_fpath)
        if step is not None:
            last_fpath = os.path.join(self.log_dir, f"checkpoint_{step:010}.pt")
            torch.save(self.get_model_state(), last_fpath)
            self.last_saved_checkpoints.append(last_fpath)
            if len(self.last_saved_checkpoints) > self.keep_interval_updates:
                old_ckpt = self.last_saved_checkpoints.pop(0)
                print(f"Removing old checkpoint {old_ckpt} "
                            f"due to keep_interval_updates={self.keep_interval_updates}")
                os.remove(old_ckpt)
        if is_best:
            best_fpath = os.path.join(self.log_dir, "checkpoint_best.pt")
            shutil.copyfile(last_fpath, best_fpath)

    def load_checkpoint(self, checkpoint_path, continue_training=True):
        print(f"Loading checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if continue_training:
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']

    def step(self, batch_x, batch_targets, training=True):
        batch_x = batch_x.to(self.device)
        batch_targets = batch_targets.to(self.device)

        batch_predicted = self.model(batch_x)

        self.optimizer.zero_grad()
        step_loss = self.criterion(batch_predicted.squeeze(1), batch_targets)

        if training:
            step_loss.backward()
            self.current_step += 1
            self.optimizer.step()

        step_loss = step_loss.detach().item()

        # batch_predicted: torch.Size([B, 4])
        batch_preds = torch.argmax(
            torch.nn.functional.softmax(batch_predicted, dim=-1),
            dim=-1
        ).cpu()

        batch_targets = batch_targets.cpu()
        step_corrects = sum(batch_targets == batch_preds)
        step_acc = step_corrects/batch_targets.size()[0]

        return step_loss, step_acc, batch_preds

    def evaluation_step(self):
        print(f"Starting evaluation")
        self.model.eval()

        eval_step_it = tqdm(
            enumerate(self.valid_data_loader),
            unit="step",
            total=len(self.valid_data_loader)
        )

        epoch_loss = []
        epoch_acc = []

        for idx, (batch_x, batch_targets) in eval_step_it:

            step_loss, step_acc, batch_preds = self.step(batch_x, batch_targets, training=False)
            epoch_loss.append(step_loss)
            epoch_acc.append(step_acc)
            loss_avg = sum(epoch_loss)/len(epoch_loss)
            acc_avg = sum(epoch_acc)/len(epoch_acc)

            log = [
                f"step: {self.current_epoch}.{idx+1}",
                f"lr: {float(self.optimizer.param_groups[0]['lr']):.2e}",
                f"avg_loss: {loss_avg:.5f}",
                f"avg_acc: {acc_avg:.5f}",
            ]

            eval_step_it.set_description(log[0])
            eval_step_it.set_postfix_str(', '.join(log[1:]))

            if idx % self.log_interval_updates == 0:
                log_interval_update = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_step}",
                    "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                    "avg_loss" : f"{loss_avg:.5f}",
                    "avg_acc" : f"{acc_avg:.5f}"
                }
                print(log_interval_update)

            if idx == 0:
                print(f"{'-'*255}"
                    "\nExamples",
                    "Target", batch_targets,
                    "Predicted", batch_preds,
                    sep="\n", end=f"\n{'-'*255}\n")

        epoch_loss_avg = sum(epoch_loss)/len(epoch_loss)
        epoch_acc_avg = sum(epoch_acc)/len(epoch_acc)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_loss_avg)

        if self.best_loss is None:
            self.best_loss = epoch_loss_avg
        if epoch_loss_avg < self.best_loss:
            self.best_loss = epoch_loss_avg
            print(f"Best loss reached: {self.best_loss}")
            self.save_checkpoint(is_best=True)

        self.tensorboard.valid_log_epoch("lr",
                                         float(self.optimizer.param_groups[0]['lr']),
                                         self.current_epoch)
        self.tensorboard.valid_log_epoch("loss", epoch_loss_avg, self.current_epoch)
        self.tensorboard.valid_log_epoch("accuracy", epoch_acc_avg, self.current_epoch)
        return epoch_loss_avg, epoch_acc_avg

    def train_one_epoch(self):
        self.model.train()

        epoch_loss = [0]
        epoch_acc = [0]
        loss_avg = acc_avg = 0

        train_step_it = tqdm(
            enumerate(self.train_data_loader),
            unit="step",
            total=len(self.train_data_loader)
        )

        for idx, (batch_x, batch_targets) in train_step_it:

            step_loss, step_acc, batch_preds = self.step(batch_x, batch_targets)
            self.tensorboard.train_log_step("loss", float(step_loss), self.current_step)

            epoch_loss.append(step_loss)
            epoch_acc.append(step_acc)

            log = [
                f"step: {self.current_epoch}.{idx+1}",
                f"lr: {float(self.optimizer.param_groups[0]['lr']):.2e}",
                f"avg_loss: {loss_avg:.5f}",
                f"avg_acc: {acc_avg:.5f}",
            ]
            train_step_it.set_description(log[0])
            train_step_it.set_postfix_str(', '.join(log[1:]))

            if idx % self.log_interval_updates == 0:
                loss_avg = sum(epoch_loss)/len(epoch_loss)
                acc_avg = sum(epoch_acc)/len(epoch_acc)
                log_interval_update = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_step}",
                    "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                    "avg_loss" : f"{loss_avg:.5f}",
                    "avg_acc" : f"{acc_avg:.5f}"
                }
                print(log_interval_update)

            if self.save_interval_updates is not None and self.current_step % self.save_interval_updates == 0:
                self.save_checkpoint(step=self.current_step)

            if self.max_updates is not None and self.current_step > self.max_updates:
                print(f"Stopping training due to max_updates={self.max_updates}")
                self.save_checkpoint()
                self.evaluation_step()

        epoch_loss_avg = sum(epoch_loss)/len(epoch_loss)
        epoch_acc_avg = sum(epoch_acc)/len(epoch_acc)

        self.tensorboard.train_log_epoch("lr",
                                        float(self.optimizer.param_groups[0]['lr']),
                                        self.current_epoch)
        self.tensorboard.train_log_epoch("loss", epoch_loss_avg, self.current_epoch)
        self.tensorboard.train_log_epoch("accuracy", epoch_acc_avg, self.current_epoch)
        return epoch_loss_avg, epoch_acc_avg

    def train(self):
        print(f"Start training")
        self.model.train()
        self.criterion = nn.CrossEntropyLoss()

        for _ in range(1, self.max_epochs+1):
            self.current_epoch += 1

            if self.current_epoch > self.max_epochs:
                print(f"Stopping training due to max_epochs={self.max_epochs}")
                self.save_checkpoint()
                valid_loss, valid_acc_string_epoch, valid_acc_epoch = self.evaluation_step()
                epoch_info = {
                    "valid_loss" : f"{valid_loss:.5f}",
                    "valid_acc" : f"{valid_acc_epoch:.5f}",
                    "valid_strings_acc" : f"{valid_acc_string_epoch:.5f}"
                }
                break

            print(f"Training epoch {self.current_epoch}")
            train_loss, train_acc_epoch = self.train_one_epoch()
            valid_loss, valid_acc_epoch = self.evaluation_step()
            epoch_info = {
                "epoch": f"{self.current_epoch}",
                "updates": f"{self.current_step}",
                "train_loss" : f"{train_loss:.5f}",
                "train_acc" : f"{train_acc_epoch:.5f}",
                "valid_loss" : f"{valid_loss:.5f}",
                "valid_acc" : f"{valid_acc_epoch:.5f}",
            }
            print(f"Epoch {self.current_epoch} done: {epoch_info}")

        print("Training done.")