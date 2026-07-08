import torch
import torch.nn.functional as F
from python.network import AlphaZeroNetwork
from python.train import AlphaZeroTrainer


def test_train_loop_decreases_loss():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = AlphaZeroNetwork(
        input_channels=1,
        height=6,
        width=7,
        num_residual_blocks=2,
        action_size=7,
        num_filters=32,
    ).to(device)

    # We must construct a dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # We must construct a dummy ReplayBuffer that provides sample()
    class DummyReplayBuffer:
        def __init__(self):
            self.states = torch.randn(64, 1, 6, 7).to(device)
            self.policies = torch.softmax(torch.randn(64, 7), dim=1).to(device)
            self.values = torch.randn(64).clamp(-1.0, 1.0).to(device)
            self.size = 64

        def sample(self, batch_size):
            # We ignore batch_size and just return our full dummy batch
            return self.states.clone(), self.policies.clone(), self.values.clone()

        def get_sampler(self):
            # Real ReplayBuffer.get_sampler() returns an RAII handle whose
            # sample() is what actually pulls data out; this dummy has no
            # sparse-to-dense cache to manage, so it can just hand back itself.
            # AlphaZeroTrainer.train() uses this as a context manager
            # (`with ... as sampler:`), so it needs __enter__/__exit__ too.
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    buffer = DummyReplayBuffer()

    trainer = AlphaZeroTrainer(
        model=model,
        replay_buffer=buffer,
        optimizer=optimizer,
        device=device,
        minibatch_size=64,
    )

    # We'll compute the loss manually before training to get a baseline
    def compute_loss():
        model.eval()
        with torch.no_grad():
            p_logits, v_preds = model(buffer.states)
            v_preds = v_preds.squeeze(-1)
            logp = F.log_softmax(p_logits, dim=1)
            policy_loss = -(buffer.policies * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(v_preds, buffer.values)
            return (policy_loss + value_loss).item()

    initial_loss = compute_loss()
    print(f"Initial loss: {initial_loss}")

    # Now we call the ACTUAL trainer method to ensure its internal loop is correct
    trainer.train(batch_size=16, train_steps=10)

    final_loss = compute_loss()
    print(f"Final loss: {final_loss}")

    assert final_loss < initial_loss, (
        f"Loss did not decrease after calling trainer.train(): started at {initial_loss}, ended at {final_loss}"
    )
