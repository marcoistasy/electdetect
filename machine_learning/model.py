import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class CNNGRU(pl.LightningModule):

    def __init__(self, output_classes, frequency_bands, input_filters=256):
        # init

        # inherit init parameters from nn.Module
        super().__init__()

        # save hyper-parameters
        self.save_hyperparameters()

        # register buffer in case of multi-gpu training
        self.register_buffer("sigma", torch.eye(3))

        # instantiate metric attributes
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(dist_sync_on_step=True),
            torchmetrics.Precision(output_classes, average='micro', dist_sync_on_step=True),
            torchmetrics.Recall(output_classes, average='micro', dist_sync_on_step=True)])
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

        # LAYERS
        # convolution layer with batch normalisation attached
        self.conv = nn.Conv2d(1, input_filters, kernel_size=(frequency_bands, 7), bias=False)
        self.conv_bn = nn.BatchNorm2d(input_filters)

        # gru layer
        self.gru = nn.GRU(input_size=input_filters, hidden_size=128, num_layers=2, batch_first=True,
                          bidirectional=False)

        # fully connected linear output layer
        self.fc = nn.Linear(128, output_classes)

    def forward(self, x):
        # activation functions for each layer

        # rectified linear unit for convolution layer
        x = F.relu(self.conv_bn(self.conv(x)))
        x = x.squeeze().permute(0, 2, 1)

        # rectified linear unit for lstm
        x = F.relu(self.gru(x)[0])
        x = F.dropout(x, p=0.5, training=self.training)  # dropout 50%

        # softmax for last layer
        x = F.softmax(self.fc(x), dim=2)

        return x

    # TRAINING

    def training_step(self, batch, batch_idx):
        # get input and target data and convert datatype
        x, y = batch
        x, y = x.float(), y.long()

        # forward pass
        y_prediction = self(x)
        y_prediction_final = y_prediction[:, -1,
                             :]  # model structure allows to visualise probability for each time step in the STFT, so only get the final probability
        loss = F.cross_entropy(input=y_prediction_final, target=y)

        # log batch metric
        outputs = self.train_metrics(y_prediction_final, y)
        outputs['Loss'] = loss
        self.log_dict(CNNGRU.metrics_dictionary(outputs, True), on_step=True, on_epoch=False, sync_dist=True)

        # return loss for back propagation
        return loss

    def training_epoch_end(self, outputs):
        # log epoch metric
        self.train_metrics.compute()

    # VALIDATION

    def validation_step(self, batch, batch_idx):
        # get input and target data and convert datatype
        x, y = batch
        x, y = x.float(), y.long()

        # forward pass
        y_prediction = self(x)
        y_prediction_final = y_prediction[:, -1,
                             :]  # model structure allows to visualise probability for each time step in the STFT, so only get the final probability
        loss = F.cross_entropy(input=y_prediction[:, -1, :], target=y)

        # log batch metric
        outputs = self.val_metrics(y_prediction_final, y)
        outputs['Loss'] = loss
        self.log_dict(CNNGRU.metrics_dictionary(outputs, False), on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        # log epoch metric
        self.val_metrics.compute()

    # TEST

    def test_step(self, batch, batch_idx):
        # get input and target data and convert datatype
        x, y = batch
        x, y = x.float(), y.long()

        # forward pass
        y_prediction = self(x)
        loss = F.cross_entropy(input=y_prediction[:, -1, :], target=y)

        return loss

    # OTHER

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    @staticmethod
    def metrics_dictionary(m, training=True):
        metrics = ['Loss', 'Accuracy', 'Precision', 'Recall']
        keys = ['{}/train'.format(i) for i in metrics] if training else ['{}/val'.format(i) for i in metrics]

        output = {}

        for i, metric in enumerate(metrics):
            output[keys[i]] = m[metric]

        return output
