from model import GCNN as TheModel
from train import train_function as the_trainer
from predict import predict_model as the_predictor
from dataset import GalaxyImageDataset as TheDataset
from dataset import GalaxyLoader as the_dataloader
from config import batchsize as the_batch_size
from config import epochs as total_epochs