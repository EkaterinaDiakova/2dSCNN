
import errno
import logging
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sparch.dataloaders.nonspiking_datasets import load_hd
from sparch.dataloaders.nonspiking_datasets import load_rusc

from sparch.models.anns_conv import CNN
from sparch.models.snns_conv import SpikeCNN
from sparch.parsers.model_config import print_model_options
from sparch.parsers.training_config import print_training_options



class Experiment:

    def __init__(self, args):

        # New model config
        self.model_type = args.model_type
        self.pdrop = args.pdrop
        # Training config
        self.use_pretrained_model = args.use_pretrained_model
        self.only_do_testing = args.only_do_testing
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.log_tofile = args.log_tofile
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.use_regularizers = args.use_regularizers
        self.reg_factor = args.reg_factor
        self.reg_fmin = args.reg_fmin
        self.reg_fmax = args.reg_fmax
        self.use_augm = args.use_augm

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()
        print_model_options(args)
        print_training_options(args)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr, weight_decay= 1e-3)

        # Define learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.opt,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """

        if not self.only_do_testing:

            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc, best_loss = self.valid_one_epoch(self.start_epoch, 0, 0)
            else:
                best_epoch, best_acc, best_loss = 0, 0, 10

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            patience = 20
            for e in range(best_epoch + 1, best_epoch + 200):
                self.train_one_epoch(e)
                best_epoch, best_acc, valid_loss = self.valid_one_epoch(e, best_epoch, best_acc)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience = 20  # Reset patience counter
                else:
                    patience -= 1
                    if patience == 0:
                        break

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")

            logging.info("\n------ Training finished ------\n")

            # Loading best model
            if self.save_best:
                if self.use_pretrained_model:
                    self.net = torch.load(
                        f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                    )
                    logging.info(
                        f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                    )
                else:
                    self.net = torch.load(
                        f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                    )
                    logging.info(
                        f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                    )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        np.save(f"{self.checkpoint_dir}/best_model_pretrained.npy", self.net.state_dict())
        if self.dataset_name in ["sc", "rusc"]:
            self.test_one_epoch(self.test_loader)
        else:
            self.test_one_epoch(self.valid_loader)
            logging.info(
                "\nThis dataset uses the same split for validation and testing.\n"
            )


    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )

        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = self.dataset_name + "_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_bdir" if self.bidirectional else "_udir"
            outname += "_reg" if self.use_regularizers else "_noreg"
            outname += "_lr" + str(self.lr)
            exp_folder = "exp/test_exps/" + outname.replace(".", "_")

        # For a new model check that out path does not exist
        if not self.use_pretrained_model and os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log_tofile:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        # For the non-spiking datasets
        if self.dataset_name in ["hd", "rusc"]:

            if self.dataset_name == "hd":
                if self.dataset_name == "hd":
                    self.frames = 135
                    self.bands = 80
                    self.nb_outputs = 20

                self.train_loader = load_hd(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="train",
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                self.valid_loader = load_hd(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="valid",
                    batch_size=self.batch_size,
                    shuffle=False,
                )

            elif self.dataset_name == "rusc":
                self.frames = 98
                self.bands = 80
                self.nb_outputs = 26

                self.train_loader = load_rusc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="train",
                    batch_size=self.batch_size,
                    shuffle=True,
                    workers=0)

                self.valid_loader = load_rusc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="valid",
                    batch_size=self.batch_size,
                    shuffle=True,
                    workers=0)

                self.test_loader = load_rusc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    shuffle=True,
                    workers=0)

        else:
            raise ValueError(f"Invalid dataset name {self.dataset_name}")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """

        if self.use_pretrained_model:

            self.pre_net = torch.load(self.load_path, map_location=self.device)
            pre_state = self.pre_net.state_dict()
            print(pre_state.keys())

            if(self.model_type == "2dCNN"):

                self.net = CNN(frames=self.frames, bands=self.bands, n_classes=self.nb_outputs).to(self.device)
                state = self.net.state_dict()
                print(state.keys())
                for key in enumerate(pre_state.keys()):
                    if key in state.keys(): state[key] = pre_state[key]
                self.net.load_state_dict(state)

            elif(self.model_type == "2dSCNN"):
                self.net = SpikeCNN(frames=self.frames, bands=self.bands, n_classes=self.nb_outputs).to(self.device)
                state = self.net.state_dict()
                print(state.keys())

                state['snn.1.conv.weight'] = pre_state['ann.1.weight']
                state['snn.3.conv.weight'] = pre_state['ann.5.weight']
                state['snn.5.conv.weight'] = pre_state['ann.9.weight']
                state['snn.7.weight'] = pre_state['ann.12.weight']
                state['snn.8.weight'] = pre_state['ann.15.weight']

                N_ann = [2, 4, 6, 8, 10, 13, 16]
                N_snn = [1, 2, 3, 4, 5, 7, 8]

                for n_ann, n_snn in zip(N_ann, N_snn):
                    state[f"snn.{n_snn}.thresh"] = pre_state[f"ann.{n_ann}.thresh"]

                self.net.load_state_dict(state)

            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        elif self.model_type in ["2dCNN"]:
            self.net = CNN(frames=self.frames, bands=self.bands, n_classes=self.nb_outputs).to(self.device)

            logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        elif self.model_type in ["2dSCNN"]:
            self.net = SpikeCNN(frames=self.frames, bands=self.bands, n_classes=self.nb_outputs).to(self.device)

            logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.model_type}")
        logging.info(f"state_dict {self.net.state_dict().keys()}")
        self.nb_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        for p in self.net.parameters():
            if p.requires_grad: logging.info(f" {p.numel()}")
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        self.net.train()
        losses, accs = [], []
        epoch_spike_rate = 0
        epoch_max_rates_for_each_layer = torch.tensor([[0., 0.]])
        # Loop over batches from train set
        for step, (x, _, y) in enumerate(self.train_loader):

            batch_start = time.time()
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass through network
            output= self.net(x)

            # Compute loss
            loss_val = self.loss_fn(output[0], y)
            losses.append(loss_val.item())

            # Backpropagate
            self.opt.zero_grad()
            loss_val.backward()
            self.opt.step()

            # Compute accuracy with labels
            pred = torch.argmax(output[0], dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)
            batch_end = time.time()
            if step == 0:
                logging.info(f"\nbatch time: , {(batch_end - batch_start)}\n")
                logging.info("\n-----------------------------\n")

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        logging.info(f"Epoch {e}: train acc={train_acc}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: train elapsed time={elapsed}")

    def valid_one_epoch(self, e, best_epoch, best_acc):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0
            epoch_max_rates_for_each_layer = torch.tensor([[0., 0.]])
            # Loop over batches from validation set
            for step, (x, _, y) in enumerate(self.valid_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output= self.net(x)

                # Compute loss
                loss_val = self.loss_fn(output[0], y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output[0], dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

            # Validation loss of whole epoch
            valid_loss = np.mean(losses)
            logging.info(f"Epoch {e}: valid loss={valid_loss}")

            # Validation accuracy of whole epoch
            valid_acc = np.mean(accs)
            logging.info(f"Epoch {e}: valid acc={valid_acc}")

            # Update learning rate
            self.scheduler.step(valid_acc)

            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e

                # Save best model
                if self.save_best:
                    torch.save(self.net, f"{self.checkpoint_dir}/best_model.pth")
                    logging.info(f"\nBest model saved with valid acc={valid_acc}")

            logging.info("\n-----------------------------\n")

            return best_epoch, best_acc, valid_loss

    def test_one_epoch(self, test_loader):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0
            epoch_max_rates_for_each_layer = torch.tensor([[0., 0.]])

            logging.info("\n------ Begin Testing ------\n")

            # Loop over batches from test set
            for step, (x, _, y) in enumerate(test_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output = self.net(x)

                # Compute
                loss_val = self.loss_fn(output[0], y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output[0], dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

            # Test loss
            test_loss = np.mean(losses)
            logging.info(f"Test loss={test_loss}")

            # Test accuracy
            test_acc = np.mean(accs)
            logging.info(f"Test acc={test_acc}")


            logging.info("\n-----------------------------\n")
