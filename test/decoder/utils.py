import torch
from spenderq.spenderq import SpenderQ


# def open_model():
#     sq = SpenderQ("qso.dr1.hiz")
#     model = sq.models[-1][0]     # the .i3 network

#     print(type(model))
#     print(model)  
#     return model

# def compare_iterations():
#     sq = SpenderQ("qso.dr1.hiz")
#     print("number of entries in sq.models:", len(sq.models))

#     ref = sq.models[-1][0].decoder.mlp[0].weight.detach().cpu()
#     for k, entry in enumerate(sq.models):
#         w = entry[0].decoder.mlp[0].weight.detach().cpu()
#         print(k, "max |difference| from last entry:", (w - ref).abs().max().item())

# def main():
#     for name in [".i0.pt", ".i1.pt", ".i2.pt", ".i3.pt", ".pt"]:
#         path = f"/work/11161/kanyuni/ls6/quassiQ_project/spenderq/src/spenderq/dat/qso.dr1.hiz{name}"   # add the directory these files live in
#         ckpt = torch.load(path, map_location="cpu", weights_only=False)
#         print(name, type(ckpt))
#         if isinstance(ckpt, dict):
#             print("  keys:", list(ckpt.keys())[:10])

# if __name__ == "__main__":
#     main()
#     #open_model()
#     compare_iterations()

dat_dir = "/work/11161/kanyuni/ls6/quassiQ_project/spenderq/src/spenderq/dat"
sq = SpenderQ("qso.dr1.hiz")

for name in [".i0.pt", ".i1.pt", ".i2.pt", ".i3.pt", ".pt"]:
    ckpt = torch.load(f"{dat_dir}/qso.dr1.hiz{name}", map_location="cpu", weights_only=False)
    models = ckpt["model"]
    print(name, "number of models in file:", len(models))

    sd = models[0]    # first model in the file, same as the [0] in sq.models[-1][0]
    key = [k for k in sd if k.endswith("decoder.mlp.0.weight")][0]

    for k, entry in enumerate(sq.models):
        w = entry[0].decoder.mlp[0].weight.detach().cpu()
        if (w - sd[key]).abs().max() == 0:
            print("   ->", "sq.models index", k)