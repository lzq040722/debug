import torch
import torch.nn as nn
import tinycudann as tcnn
from torch.optim.lr_scheduler import StepLR

class HashEncoderMotionModel(nn.Module):
    def __init__(self, bound=1.0, output_dim=3):
        super().__init__()
        self.bound = bound

        self.encoder = tcnn.Encoding(
            n_input_dims=3,


            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 4,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5
            }
        )

        enc_dim = self.encoder.n_output_dims


        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )


    def forward(self, x):

        if hasattr(self, "center") and hasattr(self, "bound_xyz"):
            x = (x - self.center) / self.bound_xyz
            x = x.clamp(-1.0, 1.0)
            x = (x + 1.0) * 0.5
        else:
            x = (x / self.bound).clamp(-1.0, 1.0)
            x = (x + 1.0) * 0.5

        feat = self.encoder(x)
        feat = feat.to(next(self.mlp.parameters()).dtype)
        return self.mlp(feat)

def build_hash_motion(pc, means3D, scene_flow, grid_res=128, grid_bounds=[-1, 1], scheduler_gamma=0.05, scheduler_step=500,
                      num_iters=400, lr=1e-4):
    with torch.enable_grad():
        assert means3D.shape[0] == scene_flow.shape[0]
        device = means3D.device
        bound = means3D.abs().max().item()


        pos_all = means3D.detach().clone().to(device).requires_grad_(True)
        flow_all = scene_flow.detach().clone().to(device).requires_grad_(False)


        model = HashEncoderMotionModel(bound=bound).to(device)
        model.train()


        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


        for step in range(num_iters):
            optimizer.zero_grad()
            pred = model(pos_all)
            loss = ((pred - flow_all) ** 2).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 50 == 0 or step == num_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"[{step:04d}/{num_iters}] Loss: {loss.item():.6f} | LR: {current_lr:.6e}")

        model.eval()
        return model
