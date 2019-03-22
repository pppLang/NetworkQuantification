import torch
from utils import updataConvWei


def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None, current_lr=None, mode=None):
    num = len(train_loader)
    model.train()
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda()
        label = label.cuda().long()
        # print(data.shape)
        result = model(data)
        loss = criterion(result, label)
        loss.backward()
        if mode=='Bin':
            updataConvWei(model, current_lr)
            # print('update weight')
        optimizer.step()
        if i%10==0:
            print('epoch {}, [{}/{}], loss {}'.format(epoch, i, num, loss))
            if writer is not None:
                writer.add_scalar('loss', loss.item(), epoch*num + i)

def test(model, test_loader, criterion, epoch, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.cuda()
            label = label.cuda()
            result = model(data)
            test_loss += criterion(result, label).item()
            pred = result.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print('epoch {}, test loss {}, acc [{}/{}]'.format(epoch, test_loss, correct, len(test_loader.dataset)))
    if writer is not None:
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('acc', correct/len(test_loader.dataset), epoch)