# %%

# imports

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from dataset import Dataset
from model import CNNGRU

# parameters

nfff = 250  # number of frequency bands to keep
nworkers = 64
nclasses = 2  # number of output classes
batch_size = 512  # batch size for training and testing
val_size = 0.3
data_path = ''  # path to dataset

# create datasets
train, val, _ = Dataset(nfff, data_path).create_data_loaders(val_size, batch_size, nworkers, 7, True)

# create a model object and logger
model = CNNGRU(nclasses, nfff)
logger = TensorBoardLogger('out/')
checkpoint_callback = ModelCheckpoint(monitor='Loss/val',
                                      save_top_k=5,
                                      dirpath='out/models/',
                                      filename='{epoch:02d}-{val_loss:.2f}',
                                      mode='min')

# initialize a trainer
trainer = pl.Trainer(gpus=8, num_nodes=1, accelerator='ddp', max_epochs=10,
                     default_root_dir='out/',
                     progress_bar_refresh_rate=20,
                     check_val_every_n_epoch=1,
                     logger=logger,
                     callbacks=[checkpoint_callback],
                     replace_sampler_ddp=False,
                     plugins=[DDPPlugin(find_unused_parameters=False)])

trainer.fit(model, train, val)
