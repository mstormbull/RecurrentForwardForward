import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import RMSprop, Adam, Adadelta
import wandb

from RecurrentFF.util import (
    Activations,
    ForwardMode,
    layer_activations_to_badness,
    standardize_layer_activations,
)
from RecurrentFF.settings import (
    Settings,
)


def amplified_initialization(layer: nn.Linear, amplification_factor=3.0):
    """Amplified initialization for Linear layers."""
    # Get the number of input features
    n = layer.in_features
    # Compute the standard deviation for He initialization
    std = (2.0 / n) ** 0.5
    # Amplify the standard deviation
    amplified_std = std * amplification_factor
    # Initialize weights with amplified standard deviation
    nn.init.normal_(layer.weight, mean=0, std=amplified_std)


class LayerMetrics:
    def __init__(self, num_layers: int):
        self.pos_activations_norms = [0 for _ in range(0, num_layers)]
        self.neg_activations_norms = [0 for _ in range(0, num_layers)]
        self.forward_weights_norms = [0 for _ in range(0, num_layers)]
        self.forward_grads_norms = [0 for _ in range(0, num_layers)]
        self.backward_weights_norms = [0 for _ in range(0, num_layers)]
        self.backward_grads_norms = [0 for _ in range(0, num_layers)]
        self.lateral_weights_norms = [0 for _ in range(0, num_layers)]
        self.lateral_grads_norms = [0 for _ in range(0, num_layers)]
        self.losses_per_layer = [0 for _ in range(0, num_layers)]

        self.num_data_points = 0

        self.update_norms = {}
        self.momentum_norms = {}
        self.update_angles = {}

    def ingest_optimizer_metrics(self, layer_num: int, forward_mode: ForwardMode, layer):
        forward_mode = str(forward_mode)

        print("=======ingesting optimizer metrics")

        for group in layer.optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    print("----problem")

                if param.grad is not None:
                    print("---- not none")
                    # compute update norm
                    update = -group['lr'] * param.grad
                    update_norm = torch.norm(update, p=2)
                    param_name = layer.param_name_dict[param]

                    if layer_num not in self.update_norms:
                        self.update_norms[layer_num] = {}
                    if forward_mode not in self.update_norms[layer_num]:
                        self.update_norms[layer_num][forward_mode] = {}
                    if param_name not in self.update_norms[layer_num]:
                        self.update_norms[layer_num][forward_mode][param_name] = 0

                    print("-------adding to update norms")
                    self.update_norms[layer_num][forward_mode][param_name] += update_norm

                    # compute momentum norm
                    try:
                        momentum_norm = torch.norm(
                            layer.optimizer.state[param]['momentum_buffer'])
                        if layer_num not in self.momentum_norms:
                            self.momentum_norms[layer_num] = {}
                        if forward_mode not in self.momentum_norms[layer_num]:
                            self.momentum_norms[layer_num][forward_mode] = {}
                        if param_name not in self.momentum_norms[layer_num]:
                            self.momentum_norms[layer_num][forward_mode][param_name] = 0

                        self.momentum_norms[layer_num][forward_mode][param_name] += momentum_norm
                    except (KeyError, AttributeError):
                        logging.debug(
                            "No momentum buffer for param. Assume using non-momentum optimizer.")

                    # compute angle
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        param.grad.view(-1), update.view(-1), dim=0)
                    cosine_similarity = torch.clamp(cosine_similarity, -1, 1)
                    angle_in_degrees = torch.acos(
                        cosine_similarity) * (180 / math.pi)
                    if layer_num not in self.update_angles:
                        self.update_angles[layer_num] = {}
                    if forward_mode not in self.update_angles[layer_num]:
                        self.update_angles[layer_num][forward_mode] = {}
                    if param_name not in self.update_angles[layer_num]:
                        self.update_angles[layer_num][forward_mode][param_name] = 0

                    self.update_angles[layer_num][forward_mode][param_name] += angle_in_degrees

    def ingest_layer_metrics(self, layer_num: int, layer, loss: int):
        pos_activations_norm = torch.norm(layer.pos_activations.current, p=2)
        neg_activations_norm = torch.norm(layer.neg_activations.current, p=2)
        forward_weights_norm = torch.norm(layer.forward_linear.weight, p=2)
        backward_weights_norm = torch.norm(layer.backward_linear.weight, p=2)
        lateral_weights_norm = torch.norm(layer.lateral_linear.weight, p=2)

        forward_grad_norm = torch.norm(layer.forward_linear.weight.grad, p=2)
        backward_grads_norm = torch.norm(
            layer.backward_linear.weight.grad, p=2)
        lateral_grads_norm = torch.norm(layer.lateral_linear.weight.grad, p=2)

        self.pos_activations_norms[layer_num] += pos_activations_norm
        self.neg_activations_norms[layer_num] += neg_activations_norm
        self.forward_weights_norms[layer_num] += forward_weights_norm
        self.forward_grads_norms[layer_num] += forward_grad_norm
        self.backward_weights_norms[layer_num] += backward_weights_norm
        self.backward_grads_norms[layer_num] += backward_grads_norm
        self.lateral_weights_norms[layer_num] += lateral_weights_norm
        self.lateral_grads_norms[layer_num] += lateral_grads_norm
        self.losses_per_layer[layer_num] += loss

    def increment_samples_seen(self):
        self.num_data_points += 1

    def average_layer_loss(self):
        return sum(self.losses_per_layer) / self.num_data_points

    def log_metrics(self, epoch):
        for i in range(0, len(self.pos_activations_norms)):
            layer_num = i+1

            metric_name = "pos_activations_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.pos_activations_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "neg_activations_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.neg_activations_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "forward_weights_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.forward_weights_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "forward_grad_norms (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.forward_grads_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "backward_weights_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.backward_weights_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "backward_grad_norms (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.backward_grads_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "lateral_weights_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.lateral_weights_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "lateral_grad_norms (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.lateral_grads_norms[i] / self.num_data_points}, step=epoch)

            metric_name = "loss (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.losses_per_layer[i] / self.num_data_points}, step=epoch)

        print(self.update_norms)
        for layer in self.update_norms:
            layer_display = str(layer + 1)
            print("layer: " + str(layer))

            for forward_mode in self.update_norms[layer]:
                print("forward mode: " + str(forward_mode))
                forward_mode_display = self.__forward_mode_display(
                    forward_mode)

                for param_name in self.update_norms[layer][forward_mode]:
                    print("--------update went through")
                    metric_name = f"{forward_mode_display} {param_name} update norm (layer {layer_display})"
                    wandb.log(
                        {metric_name: self.update_norms[layer][param_name] / self.num_data_points}, step=epoch)

        for layer in self.momentum_norms:
            layer_display = str(layer + 1)

            for forward_mode in self.momentum_norms[layer]:
                forward_mode_display = self.__forward_mode_display(
                    forward_mode)

                for param_name in self.momentum_norms[layer][forward_mode]:
                    metric_name = f"{forward_mode_display} {param_name} momentum (layer {layer_display})"
                    wandb.log(
                        {metric_name: self.momentum_norms[layer][forward_mode][param_name] / self.num_data_points}, step=epoch)

        for layer in self.update_angles:
            layer_display = str(layer + 1)

            for forward_mode in self.update_angles[layer]:
                forward_mode_display = self.__forward_mode_display(
                    forward_mode)

                for param_name in self.update_angles[layer][forward_mode]:
                    metric_name = f"{forward_mode_display} update angle (layer {layer_display})"
                    wandb.log(
                        {metric_name: self.update_angles[layer][forward_mode][param_name] / self.num_data_points}, step=epoch)

    def __forward_mode_display(self, forward_mode: ForwardMode):
        if str(ForwardMode.PositiveData) == forward_mode:
            return "pos"
        elif str(ForwardMode.NegativeData) == forward_mode:
            return "neg"
        else:
            logging.error("Unexpected forward mode. Failing fast.")
            exit(1)


class HiddenLayer(nn.Module):
    """
    A HiddenLayer class for a novel Forward-Forward Recurrent Network, with
    inspiration drawn from Boltzmann Machines and Noise Contrastive Estimation.
    This network design is characterized by two distinct forward passes, each
    with specific objectives: one is dedicated to processing positive ("real")
    data with the aim of lowering the 'badness' across every hidden layer,
    while the other is tasked with processing negative data and adjusting the
    weights to increase the 'badness' metric.

    The HiddenLayer is essentially a node within this network, with possible
    connections to both preceding and succeeding layers, depending on its
    specific location within the network architecture. The first layer in this
    setup is connected directly to the input data, and the last layer maintains
    a connection to the output data. The intermediate layers establish a link to
    both their previous and next layers, if available.

    In each HiddenLayer, a forward linear transformation and a backward linear
    transformation are defined. The forward transformation is applied to the
    activations from the previous layer, while the backward transformation is
    applied to the activations of the subsequent layer. The forward
    transformation helps in propagating the data through the network, and the
    backward transformation is key in the learning process where it aids in the
    adjustment of weights based on the output or next layer's activations.
    """

    def __init__(
            self,
            settings: Settings,
            train_batch_size,
            test_batch_size,
            prev_size,
            size,
            next_size,
            damping_factor):
        super(HiddenLayer, self).__init__()

        self.settings = settings

        self.train_activations_dim = (train_batch_size, size)
        self.test_activations_dim = (test_batch_size, size)

        self.damping_factor = damping_factor

        self.pos_activations = None
        self.neg_activations = None
        self.predict_activations = None
        self.reset_activations(True)

        self.forward_linear = nn.Linear(prev_size, size)
        nn.init.kaiming_uniform_(
            self.forward_linear.weight, nonlinearity='relu')

        self.backward_linear = nn.Linear(next_size, size)

        if next_size == self.settings.data_config.num_classes:
            amplified_initialization(self.backward_linear, 3.0)
        else:
            nn.init.uniform_(self.backward_linear.weight, -0.05, 0.05)

        # Initialize the lateral weights to be the identity matrix
        self.lateral_linear = nn.Linear(size, size)
        nn.init.orthogonal_(self.lateral_linear.weight, gain=math.sqrt(2))

        self.previous_layer = None
        self.next_layer = None

        if self.settings.model.ff_optimizer == "adam":
            self.optimizer = Adam(self.parameters(),
                                  lr=self.settings.model.ff_adam.learning_rate)
        elif self.settings.model.ff_optimizer == "rmsprop":
            self.optimizer = RMSprop(
                self.parameters(),
                lr=self.settings.model.ff_rmsprop.learning_rate,
                momentum=self.settings.model.ff_rmsprop.momentum)
        elif self.settings.model.ff_optimizer == "adadelta":
            self.optimizer = Adadelta(
                self.parameters(),
                lr=self.settings.model.ff_adadelta.learning_rate)

        self.param_name_dict = {param: name for name,
                                param in self.named_parameters()}

    def _apply(self, fn):
        """
        Override apply, but we don't want to apply to sibling layers as that
        will cause a stack overflow. The hidden layers are contained in a
        collection in the higher-level RecurrentFFNet. They will all get the
        apply call from there.
        """
        # Apply `fn` to each parameter and buffer of this layer
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        # Then remove `previous_layer` and `next_layer` temporarily
        previous_layer = self.previous_layer
        next_layer = self.next_layer
        self.previous_layer = None
        self.next_layer = None

        # Apply `fn` to submodules
        for module in self.children():
            module._apply(fn)

        # Restore `previous_layer` and `next_layer`
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        return self

    def reset_activations(self, isTraining):
        activations_dim = None
        if isTraining:
            activations_dim = self.train_activations_dim

            pos_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            pos_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            self.pos_activations = Activations(
                pos_activations_current, pos_activations_previous)

            neg_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            neg_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            self.neg_activations = Activations(
                neg_activations_current, neg_activations_previous)

            self.predict_activations = None

        else:
            activations_dim = self.test_activations_dim

            predict_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            predict_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            self.predict_activations = Activations(
                predict_activations_current, predict_activations_previous)

            self.pos_activations = None
            self.neg_activations = None

    def advance_stored_activations(self):
        if self.pos_activations is not None:
            self.pos_activations.advance()

        if self.neg_activations is not None:
            self.neg_activations.advance()

        if self.predict_activations is not None:
            self.predict_activations.advance()

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def train(self, input_data, label_data, should_damp, layer_metrics: LayerMetrics, layer_num: int):
        self.optimizer.zero_grad()

        pos_activations = None
        neg_activations = None
        if input_data is not None and label_data is not None:
            (pos_input, neg_input) = input_data
            (pos_labels, neg_labels) = label_data

            neg_activations = self.forward(
                ForwardMode.NegativeData, neg_input, neg_labels, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.NegativeData, self)

            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, pos_labels, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.PositiveData, self)

        elif input_data is not None:
            (pos_input, neg_input) = input_data

            neg_activations = self.forward(
                ForwardMode.NegativeData, neg_input, None, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.NegativeData, self)

            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, None, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.PositiveData, self)

        elif label_data is not None:
            (pos_labels, neg_labels) = label_data

            neg_activations = self.forward(
                ForwardMode.NegativeData, None, neg_labels, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.NegativeData, self)

            pos_activations = self.forward(
                ForwardMode.PositiveData, None, pos_labels, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.PositiveData, self)
        else:
            neg_activations = self.forward(
                ForwardMode.NegativeData, None, None, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.NegativeData, self)

            pos_activations = self.forward(
                ForwardMode.PositiveData, None, None, should_damp)
            layer_metrics.ingest_optimizer_metrics(
                layer_num, ForwardMode.PositiveData, self)

        pos_badness = layer_activations_to_badness(pos_activations)
        neg_badness = layer_activations_to_badness(neg_activations)

        # Loss function equivelent to:
        # plot3d log(1 + exp(-n + 1)) + log(1 + exp(p - 1)) for n=0 to 3, p=0 to 3
        layer_loss = F.softplus(torch.cat([
            (-1 * neg_badness) + self.settings.model.loss_threshold,
            pos_badness - self.settings.model.loss_threshold
        ])).mean()
        layer_loss.backward()

        self.optimizer.step()
        return layer_loss

    def forward(self, mode, data, labels, should_damp):
        """
        Propagates input data forward through the network, updating the
        activation state of the current layer based on the operating mode.

        Handles various scenarios depending on the layer configuration in the
        network (input layer, output layer, or a middle layer).

        Args:
            mode (ForwardMode enum): Indicates the type of data being propagated
            (positive, negative, or prediction).

            data (torch.Tensor or None): The input data for the layer. If
            `None`, it indicates that this layer is not the input layer.

            labels (torch.Tensor or None): The target labels for the layer. If
            `None`, it indicates that this layer is not the output layer.

            should_damp (bool): A flag to determine whether the activation
            damping should be applied.

        Returns:
            new_activation (torch.Tensor): The updated activation state of the
            layer after the forward propagation.

        Note:
            'Damping' here refers to a technique used to smoothen the changes in
            the layer activations over time. In this function, damping is
            implemented as a weighted average of the previous and the newly
            computed activations, controlled by the `self.damping_factor`.

            The function expects to receive input data and/or labels depending
            on the layer. The absence of both implies the current layer is a
            'middle' layer. If only the labels are missing, this layer is an
            'input' layer, while if only the data is missing, it's an 'output'
            layer. If both are provided, the network has only a single layer.

            All four scenarios are handled separately in the function, although
            the general procedure is similar: compute new activations based on
            the received inputs (and possibly, depending on the layer's
            position, the activations of the adjacent layers), optionally apply
            damping, update the current layer's activations, and return the new
            activations.
        """
        # Make sure assumptions aren't violated regarding layer connectivity.
        if data is None:
            assert self.previous_layer is not None
        if labels is None:
            assert self.next_layer is not None

        # Middle layer.
        new_activation = None
        if data is None and labels is None:
            next_layer_prev_timestep_activations = None
            prev_layer_prev_timestep_activations = None
            prev_act = None
            if mode == ForwardMode.PositiveData:
                next_layer_prev_timestep_activations = self.next_layer.pos_activations.previous
                prev_layer_prev_timestep_activations = self.previous_layer.pos_activations.previous
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                next_layer_prev_timestep_activations = self.next_layer.neg_activations.previous
                prev_layer_prev_timestep_activations = self.previous_layer.neg_activations.previous
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                next_layer_prev_timestep_activations = self.next_layer.predict_activations.previous
                prev_layer_prev_timestep_activations = self.previous_layer.predict_activations.previous
                prev_act = self.predict_activations.previous

            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations.detach()
            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)

            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations.detach()
            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            new_activation =  \
                F.leaky_relu(F.linear(
                    prev_layer_stdized,
                    self.forward_linear.weight)) + \
                -1 * F.leaky_relu(F.linear(
                    next_layer_stdized,
                    self.backward_linear.weight)) + \
                F.leaky_relu(F.linear(
                    prev_act_stdized,
                    self.lateral_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

        # Single layer scenario. Hidden layer connected to input layer and
        # output layer.
        elif data is not None and labels is not None:
            prev_act = None
            if mode == ForwardMode.PositiveData:
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                prev_act = self.predict_activations.previous

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            new_activation = \
                F.leaky_relu(F.linear(
                    data,
                    self.forward_linear.weight)) + \
                -1 * F.leaky_relu(F.linear(
                    labels,
                    self.backward_linear.weight)) + \
                F.leaky_relu(F.linear(
                    prev_act_stdized,
                    self.lateral_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

        # Input layer scenario. Connected to input layer and hidden layer.
        elif data is not None:
            prev_act = None
            next_layer_prev_timestep_activations = None
            if mode == ForwardMode.PositiveData:
                next_layer_prev_timestep_activations = self.next_layer.pos_activations.previous
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                next_layer_prev_timestep_activations = self.next_layer.neg_activations.previous
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                next_layer_prev_timestep_activations = self.next_layer.predict_activations.previous
                prev_act = self.predict_activations.previous

            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations.detach()
            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            new_activation = \
                F.leaky_relu(F.linear(
                    data,
                    self.forward_linear.weight)) + \
                -1 * F.leaky_relu(F.linear(
                    next_layer_stdized,
                    self.backward_linear.weight)) + \
                F.leaky_relu(F.linear(
                    prev_act_stdized,
                    self.lateral_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

        # Output layer scenario. Connected to hidden layer and output layer.
        elif labels is not None:
            prev_layer_prev_timestep_activations = None
            prev_act = None
            if mode == ForwardMode.PositiveData:
                prev_layer_prev_timestep_activations = self.previous_layer.pos_activations.previous
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                prev_layer_prev_timestep_activations = self.previous_layer.neg_activations.previous
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                prev_layer_prev_timestep_activations = self.previous_layer.predict_activations.previous
                prev_act = self.predict_activations.previous

            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations.detach()
            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            new_activation = \
                F.leaky_relu(F.linear(
                    prev_layer_stdized,
                    self.forward_linear.weight)) + \
                -1 * F.leaky_relu(F.linear(
                    labels,
                    self.backward_linear.weight)) + \
                F.leaky_relu(F.linear(
                    prev_act_stdized,
                    self.lateral_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

        if mode == ForwardMode.PositiveData:
            self.pos_activations.current = new_activation
        elif mode == ForwardMode.NegativeData:
            self.neg_activations.current = new_activation
        elif mode == ForwardMode.PredictData:
            self.predict_activations.current = new_activation

        return new_activation
