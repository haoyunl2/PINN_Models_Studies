import yaml
import torch
from torch.utils.data import DataLoader
from model import DCNN2d
from load import DarcyFlow

from test_2d import test_darcy

from argparse import ArgumentParser