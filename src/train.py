from lightning import Trainer, seed_everything

from networks import MMDiT
from datasets import Sinusoids


if __name__ == "__main__":
    seed_everything(42)
    dataset = Sinusoids(batch_size=256, n_sources=2, n_times=128, size=256)

    model = MMDiT(
        x_shape=(dataset.n_sources, dataset.n_params),
        c_shape=(dataset.n_times, dataset.n_channels),
        hidden_dim=4 * 64,
        num_heads=4,
        num_blocks=8,
    )

    trainer = Trainer(
        max_time="00:24:00:00",
        precision="16-mixed",
        gradient_clip_val=1.0,
        deterministic=True,
        enable_checkpointing=True,
        accumulate_grad_batches=1,
    )

    trainer.fit(model, dataset.dataloader())
