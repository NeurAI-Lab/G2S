# Copyright © NavInfo Europe 2021.

import torch
import math


class G2S(torch.nn.Module):
    """
    GPS-to-Scale (G2S) loss
    """
    def __init__(self, num_epochs, altitude_with_gps=True):
        """
        Initializes G2S loss

        Parameters
        ----------
        num_epochs: Total number of epochs for which the network will be trained.
        altitude_with_gps: if gps has altitude values.
        """
        super(G2S, self).__init__()
        self.num_epochs = num_epochs
        self.altitude_with_gps = altitude_with_gps

    def forward(self, gps, translation, epoch):
        """
        Computes the g2s loss

        Parameters
        ----------
        gps: The input gps to the network loaded from the GPSDataloader class.
        translation: The translation prediction made by the network of shape 12 x 3.
        epoch: Current training epoch number indexed from 0.
        """

        dis = {}
        if self.altitude_with_gps:
            dis["-1, 0"] = torch.norm(translation["-1, 0"], dim=1)
            dis["0, 1"] = torch.norm(translation["0, 1"], dim=1)
        else:
            dis["-1, 0"] = torch.norm(translation["-1, 0"][:, [0, 2]], dim=1)
            dis["0, 1"] = torch.norm(translation["0, 1"][:, [0, 2]], dim=1)

        s1 = gps["-1, 0"].float() / dis["-1, 0"]
        s2 = gps["0, 1"].float() / dis["0, 1"]

        loss = torch.mean((s1 - 1) ** 2 + (s2 - 1) ** 2)
        weight = self.inverse_exp_epoch_weight(epoch)

        g2s_loss = weight * loss
        scale = 0.5 * torch.mean(s1 + s2)

        return {
            'g2s_loss': g2s_loss,
            'scale': scale
        }

    def inverse_exp_epoch_weight(self, epoch):
        """
        Computes the dynamic weight for the G2S loss

        Parameters
        ----------
        epoch: The training epoch. The weight increases exponentially with the epochs.
        """
        return math.exp(epoch - self.num_epochs + 1)

