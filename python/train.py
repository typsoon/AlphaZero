import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable, Type


from network import AlphaZeroNetwork

import sys
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from pybind import self_play_bind, engine_bind
from self_play_bind import self_play  # pyright: ignore
from engine_bind import Game, ReplayBuffer  # pyright: ignore


tqdm = notebook_tqdm if "ipykernel" in sys.modules else base_tqdm


class AlphaZeroTrainer:
    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        minibatch_size=4096,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.device = device
        assert (
            torch.device(device).type == next(model.parameters()).device.type
            # and device.index == next(model.parameters()).device.index
        ), (
            f"device par: {device} should be the same as {next(model.parameters()).device}"
        )

    def train(self, batch_size=64, train_steps=1000):
        self.model.train()
        # torch.autograd.detect_anomaly()
        accum_steps = self.minibatch_size // batch_size
        assert self.minibatch_size % batch_size == 0
        # print(self.replay_buffer.size, self.minibatch_size)

        progress_bar: Any = range(train_steps)

        policy_loss: torch.Tensor = torch.tensor(0.0)
        value_loss: torch.Tensor = torch.tensor(0.0)

        if __debug__:
            progress_bar = tqdm(progress_bar)

        for step in progress_bar:
            states, target_policies, target_values = self.replay_buffer.sample(
                self.minibatch_size
            )
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            # print(states.shape, target_policies.shape, target_values.shape)
            # print(states, target_policies, target_values)
            # print(target_policies, target_values)
            if states.shape[0] < self.minibatch_size:
                raise Warning(
                    f"This shouldn't happen, {states.shape[0]}, {self.minibatch_size}"
                )  # albo raise Warning

            for i in range(accum_steps):
                start = i * batch_size
                end = start + batch_size

                s_batch = states[start:end]

                assert s_batch.shape[0] != 0, (
                    f"s_batch shouldn't be empty, {start} {end}"
                )

                pi_batch = target_policies[start:end]
                v_batch = target_values[start:end]

                s_batch = states[start:end]
                pi_batch = target_policies[start:end]
                v_batch = target_values[start:end]

                p_logits, v_preds = self.model(s_batch)
                v_preds = v_preds.squeeze()

                logp = F.log_softmax(p_logits, dim=1)
                pi_batch = pi_batch + 1e-8
                pi_batch = pi_batch / pi_batch.sum(dim=1, keepdim=True)

                # print("p_logits:", p_logits.min().item(), p_logits.max().item())
                # print("pi_batch sum:", pi_batch.sum(dim=1))
                # print("pi_batch any negative:", (pi_batch < 0).any().item())
                # print("pi_batch any zero:", (pi_batch == 0).any().item())
                assert v_preds.shape[0] != 0, f"{p_logits} {s_batch}"
                # print(
                #     v_preds,
                #     v_batch,
                #     "MSE LOSS: ",
                #     F.mse_loss(v_preds.squeeze(), v_batch),
                #     f"{value_loss.item():.4f}",
                # )
                # sleep(1)

                # print("logp any inf:", torch.isinf(logp).any().item())
                # print("logp any NaN:", torch.isnan(logp).any().item())

                policy_loss = F.kl_div(logp, pi_batch, reduction="batchmean")
                value_loss = F.mse_loss(v_preds, v_batch)

                assert v_preds.shape == v_batch.shape, (
                    f"{v_preds.shape} {v_batch.shape}"
                )
                value_loss = F.mse_loss(v_preds, v_batch)

                loss = policy_loss + value_loss
                loss /= accum_steps
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if __debug__:
                if step % 100 == 0:
                    progress_bar.set_postfix(
                        {
                            "policy loss": f"{policy_loss.item():.4f}",  # pyright: ignore
                            "value loss": f"{value_loss.item():.4f}",  # pyright: ignore
                        }
                    )


def self_play_and_train_loop(
    network_type: Type[AlphaZeroNetwork],
    network_path: str,
    network_device: torch.device,
    game_type: Type[Game],
    trainer_factory: Callable[
        [nn.Module, torch.device, ReplayBuffer, int], AlphaZeroTrainer
    ],
    loop_iterations: int = 1,
    games_in_each_iteration: int = 100,
    batch_size=256,
    training_iterations=20,
    thread_count: int = 1,
    replay_buffer_size=1000,
    minibatch_size=4096,
):
    game = game_type()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    latest_network_path = network_path
    network = network_type.load_az_network(latest_network_path, network_device)

    for _ in range(loop_iterations):
        latest_scripted_network_path = latest_network_path + "_scripted.pt"
        network.script_and_save_network(latest_scripted_network_path)

        self_play(
            game,
            latest_scripted_network_path,
            replay_buffer,
            games_in_each_iteration,
            thread_count,
        )

        trainer = trainer_factory(
            network, network_device, replay_buffer, minibatch_size
        )
        trainer.train(batch_size, training_iterations)

        latest_network_path = network_path + "_trained"
        network.save_az_network(latest_network_path)
