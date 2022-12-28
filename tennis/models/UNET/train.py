import argparse
import os

import cv2
import random
import torch
import tqdm
import time

import numpy as np

#from lib.algorithms.ball_detection_unet.data_loader import BallUNetDataset
from data_loader_tennis import BallUNetDataset
#from lib.algorithms.ball_detection_unet.unet import unet
from unet import unet

LEARNING_RATE = .00005
WEIGHT_DECAY = 0
EPOCHS = 1

# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)

def distillation_loss(student, teacher, gold):
    KL = teacher * torch.log((teacher + 1e-8) / (student + 1e-8)) + (1 - teacher) * torch.log((1 - teacher + 1e-8) / (1 - student + 1e-8))
    weights = gold * 9 + 1
    return (KL * weights).mean()


def train(model, device, optimizer, train_loader, checkpoint_path, clip_norm=0.5, train=True, large_model=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = []
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()



        if train:
            output = model(data[:, 0:3], data[:, 0:3], data[:, 0:3])

            # Model distillation based on the larger model
            if large_model:
                with torch.no_grad():
                    teacher_label = (large_model(data[:, 0:3], data[:, 0:3], data[:, 0:3]) > 0.3).float()
                loss = distillation_loss(output, teacher_label, label)
            else:
                loss = model.weighted_binary_cross_entropy(output, label, weights=np.array([1, 5]))
            if torch.isnan(loss):
                print("skipping nan")
                import pdb
                pdb.set_trace()
                continue
            loss.backward()

            # Clip the gradient
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

        else:
            with torch.no_grad():
                output = model(data[:, 0:3], data[:, 0:3], data[:, 0:3])
                loss = model.weighted_binary_cross_entropy(output, label, weights=np.array([1, 5]))

        total_loss.append(loss.detach().cpu().numpy())

        # Check for Nan
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        if np.random.uniform() < 0.1:

            to_write = output[0, 0].detach().cpu().numpy()
            cv2.imwrite("training.png", to_write * 255)

            input_frame = data[0, 3:6].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
            cv2.imwrite("input.png", (input_frame + 1) * 127.5)

            target = label[0, 0].detach().cpu().numpy()
            cv2.imwrite("target.png", target * 255)
            
            print("Loss: {}".format(np.array(total_loss).mean()))

    print("Loss: {}".format(np.array(total_loss).mean()))
    torch.save(model.state_dict(), os.path.join(checkpoint_path, "model_1frame.pth"))



def main(args):
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    data_train = BallUNetDataset(args.data_train_path)
    data_test = BallUNetDataset(args.data_test_path)

    kwargs = {'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               **kwargs)

    model = unet().to(device)

    if args.pretrain_path:
        state_dict = torch.load(args.pretrain_path).state_dict()
        model.load_pretrain(state_dict)

    if args.distillation_model:
        large_model = torch.load(args.distillation_model, map_location=device).to(device)
        large_model.eval()
    else:
        large_model = None

    for epoch in range(EPOCHS):
        print("Epoch: {}".format(epoch))
        optimizer = torch.optim.Adam(model.parameters(), lr=(LEARNING_RATE * (0.95 ** epoch)),
                                     weight_decay=WEIGHT_DECAY)

        train_loss = train(model, device, optimizer, train_loader, args.checkpoint_path, large_model=large_model)
        test_loss = train(model, device, optimizer, test_loader, args.checkpoint_path, train=False) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for network/main.py')
    parser.add_argument("data_train_path", type=str, help="Path to training samples")
    parser.add_argument("data_test_path", type=str, help="Path to testing samples")
    parser.add_argument("checkpoint_path", type=str, help="Path to save model")
    parser.add_argument("--model-load", type=str, help="Path to starting model weights")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pretrain-path", type=str, help="Path to pretrained model")
    parser.add_argument("--distillation-model", type=str, help="Path to a larger model")
    main(parser.parse_args())
