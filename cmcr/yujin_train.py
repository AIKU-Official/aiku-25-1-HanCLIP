import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim

from cmcr.yujin_model import TriAlignModel
from cmcr.type import ModalityType


def contrastive_loss(anchor, positive, temperature=0.07):
    logits = torch.matmul(anchor, positive.T) / temperature
    labels = torch.arange(anchor.size(0)).to(anchor.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def train_one_epoch(model, dataloader, optimizer, device):
    model.projector.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_dict = {
            ModalityType.TEXT: batch['text'],
            ModalityType.VISION: batch['image']
        }

        text_feat = model.trunk.get_text_feature(input_dict[ModalityType.TEXT])
        image_feat = model.trunk.get_vision_feature(input_dict[ModalityType.VISION])

        proj_text = model.projector.forward_text(text_feat)
        proj_image = model.projector.forward_image(image_feat)

        loss_t2i = contrastive_loss(proj_text, proj_image)
        loss_i2t = contrastive_loss(proj_image, proj_text)
        loss = (loss_t2i + loss_i2t) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.projector.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_dict = {
                ModalityType.TEXT: batch['text'],
                ModalityType.VISION: batch['image']
            }

            text_feat = model.trunk.get_text_feature(input_dict[ModalityType.TEXT])
            image_feat = model.trunk.get_vision_feature(input_dict[ModalityType.VISION])

            proj_text = model.projector.forward_text(text_feat)
            proj_image = model.projector.forward_image(image_feat)

            sim = torch.matmul(proj_text, proj_image.T)
            pred = torch.argmax(sim, dim=1)
            labels = torch.arange(sim.size(0)).to(device)

            correct += (pred == labels).sum().item()
            total += sim.size(0)

    return correct / total


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TriAlignModel(device=device)

    train_set = YourMultimodalDataset(split='train')
    val_set = YourMultimodalDataset(split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    optimizer = optim.Adam(model.projector.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        torch.save(model.projector.state_dict(), f"{args.save_path}/trialign_epoch{epoch+1}.pt")


if __name__ == '__main__':
    main()
