import torch

from libreyolo.models.yolo9.drax import DraxBlock


def main():
    x = torch.randn(
        2,
        256,
        20,
        20,
    )

    block = DraxBlock(
        dim=256,
        use_attention=True,
        efficient=True,
        fusion_mode="average",
        drop_path=0.0,
    )

    block.eval()

    with torch.no_grad():
        y = block(x)

    print("input :", x.shape)
    print("output:", y.shape)

    assert x.shape == y.shape


if __name__ == "__main__":
    main()
