from lightning import Trainer, seed_everything
import torch
from networks import MMDiT
from datasets import Sinusoids


if __name__ == "__main__":
    seed_everything(42)
    dataset = Sinusoids(batch_size=256, n_sources=2, n_times=1024)
    dataloader = dataset.dataloader(num_workers=8, persistent_workers=True)

    model = MMDiT(
        x_dim=dataset.n_params,
        hidden_dim=8 * 32,
        num_heads=8,
        num_blocks=8,
    )
    model = torch.compile(model)

    trainer = Trainer(
        max_time="00:24:00:00",
        precision="16-mixed",
        gradient_clip_val=1.0,
        deterministic=True,
        enable_checkpointing=True,
        accumulate_grad_batches=1,
    )

    trainer.fit(model, dataloader)
