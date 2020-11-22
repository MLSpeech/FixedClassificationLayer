import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import argparse
from dataloader import load_stl
from tqdm import *
from resnet import resnet18

parser = argparse.ArgumentParser(description='PyTorch Fixed representations.')
parser.add_argument('--runname', type=str, default='default_name', help='the name of the run. ckpt will be saved under that name')
parser.add_argument('--save_dir', default='.', type=str, help='saving dir. model will be saved under that dir')
parser.add_argument('--data_dir', default='.', type=str, help='dir to STL dataset. If the dataset does not exist, it will be downloaded to the given dir')
parser.add_argument('--cosine', default=False, action='store_true', help='When True optimize the cosine-similarity. Else optimize the dot-product')
parser.add_argument('--fixed', default=False, action='store_true', help='When True class vectors are fixed')
parser.add_argument('--s', default=1, type=int, help='scaling factor')
parser.add_argument('--cuda', default=1, type=int, help='cuda device id')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

EPOCHS = 80
NUM_OF_CLASSES = 10
EMBEDDING_SIZE = 4608
MODEL = 'RESNET18'
LR = 0.001 if not args.fixed and not args.cosine else 0.1

if args.s > 1 and not args.cosine:
    raise ValueError('S is supported in cosine-similarity maximization only')

def train(epoch, model, loss_func, device, loader, optimizer, args, embeddings, bias):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(loader)):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        out, out_pre_norm = model(data)

        if args.cosine:
            output = F.log_softmax(out.matmul(F.normalize(embeddings, p=2, dim=0))*args.s + bias)
        else:
            output = F.log_softmax(out_pre_norm.matmul(embeddings) + bias, dim=1)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100. * batch_idx / len(loader), loss.item()))


def evaluate(model, loss_func, device, loader, args, embeddings, bias):
    model.eval()
    loss = 0
    correct = 0
    correct_norms, wrong_norms = list(), list()
    correct_max_angle, wrong_max_angle = list(), list()

    with torch.no_grad():
        for data, target in tqdm(loader):

            data, target = data.to(device), target.to(device)

            out, out_pre_norm = model(data)

            normalized_embeddings = F.normalize(embeddings, p=2, dim=0)
            matmul = out.matmul(normalized_embeddings)
            max_angle = matmul.max(1, keepdim=True)[0].cpu().numpy().tolist()

            if args.cosine:
                output = F.log_softmax(out.matmul(normalized_embeddings) * args.s + bias)
            else:
                output = F.log_softmax(out_pre_norm.matmul(embeddings) + bias, dim=1)

            loss += loss_func(output, target).item()

            pred1 = output.max(1, keepdim=True)[1]
            is_eq_pred1 = pred1.eq(target.view_as(pred1)).cpu().numpy().tolist()

            out_pre_norm = out_pre_norm.cpu().detach().numpy()

            for idx in range(len(is_eq_pred1)):
                if is_eq_pred1[idx][0] == 1:
                    correct_max_angle.append(max_angle[idx][0])
                    correct_norms.append(np.linalg.norm(out_pre_norm[idx]))
                    correct += 1
                else:
                    wrong_max_angle.append(max_angle[idx][0])
                    wrong_norms.append(np.linalg.norm(out_pre_norm[idx]))

    loss /= len(loader.dataset)

    return loss, (100 * correct / len(loader.dataset)), correct, len(loader.dataset), \
           np.average(correct_norms), np.average(wrong_norms), np.average(correct_max_angle), np.average(wrong_max_angle)


def get_embeddings(embedding_size):
    word2vec_dictionary = dict()
    for cls_idx in range(NUM_OF_CLASSES):
        v = np.random.randint(low=-100 , high=100, size=embedding_size)
        v = v / np.linalg.norm(v)
        word2vec_dictionary[cls_idx] = torch.from_numpy(v).float()

    w2v_matrix = torch.stack(list(word2vec_dictionary.values()), dim=1)
    bias = torch.ones(NUM_OF_CLASSES) * 0.01

    return w2v_matrix.clone(), bias.clone()


def save_model(path, runname, model, epoch, loss, acc):
    ckpt_dir = os.path.join(path, '{}.ckpt'.format(runname))

    print('Saving the model to {}'.format(path))

    torch.save(dict(model_state=model.state_dict(), epoch=epoch, loss=loss, acc=acc), ckpt_dir)


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.cuda) if use_cuda else "cpu")
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    embeddings, bias = get_embeddings(EMBEDDING_SIZE)
    embeddings = embeddings.to(device)
    bias = bias.to(device)

    model = resnet18()
    model.to(device)

    loss_func = F.nll_loss

    if not args.cosine:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)

    if not args.fixed:
        embeddings = embeddings.requires_grad_(True)
        bias = bias.requires_grad_(True)
        optimizer.add_param_group({'params': embeddings})
        optimizer.add_param_group({'params': bias})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3,
                                                               verbose=True, threshold_mode='abs')

    train_loader, test_loader = load_stl(args.data_dir, args.batch_size, args.num_workers)

    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, loss_func, device, train_loader, optimizer, args, embeddings, bias)


        print('\nEpoch {} Results:'.format(epoch))
        print('embeddings at epoch-{}: {}'.format(epoch,embeddings))

        print('Evaluating on the trainig set...')
        train_loss, train_acc, train_correct, train_len, _, _, _, _ = evaluate(model, loss_func, device, train_loader, args, embeddings, bias)
        print(
            'Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(train_loss, train_correct, train_len,
                                                                               train_acc))

        print('Evaluating on the test set...')
        test_loss, test_acc, test_correct, test_len, test_avg_correct_norms, \
        test_avg_wrong_norms, test_avg_correct_angle, test_avg_wrong_angle =\
            evaluate(model, loss_func, device, test_loader, args, embeddings, bias)

        print(
            'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, test_correct, test_len,
                                                                                   test_acc))

        scheduler.step(test_acc)
        if best_acc < test_acc:
            save_model(save_dir, args.runname, model, epoch, test_loss, test_acc)
            best_acc = test_acc
            torch.save(embeddings, os.path.join(save_dir, 'embedding'))
            torch.save(bias, os.path.join(save_dir, 'bias'))

        print('# Epoch stats:')
        print('Train/Test loss:{:.3f}/{:.3f}'.format(train_loss, test_loss))
        print('Train/Test acc:{:.2f}/{:.2f}'.format(train_acc, test_acc))
        print('Best acc: {:.3f}  {}\n'.format(best_acc,LR))

        if args.fixed:
            if args.cosine:
                print('STL trained fixed cosine-similarity')
            else:
                print('STL trained fixed dot-product')
        else:
            if args.cosine:
                print('STL trained non-fixed cosine-similarity')
            else:
                print('STL trained non-fixed dot-product')
