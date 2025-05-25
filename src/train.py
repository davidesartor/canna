import torch
from lightning import Trainer, seed_everything
from networks import MMDiT
from datasets import Sinusoids

if __name__ == "__main__":
    seed_everything(42)

    dataset = Sinusoids(batch_size=1024, n_sources=2, n_times=1024)
    dataloader = dataset.dataloader(num_workers=32, persistent_workers=True)

    model = MMDiT(
        x_dim=dataset.n_params,
        hidden_dim=8 * 64,
        num_heads=8,
        num_blocks=8,
    )
    model = torch.compile(model)

    trainer = Trainer(
        max_time="00:24:00:00",
        precision="32-true",
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        accumulate_grad_batches=1,
    )

    trainer.fit(model, dataloader)
