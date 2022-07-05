import torch
from torch import nn
import datetime
import matplotlib.pyplot as plt
import time
from asteroid.losses import PITLossWrapper
from asteroid.losses import pairwise_neg_sisdr


class Trainer():
    def __init__(self, num_epochs, device, n_fft=512, lr=1e-4, loss_fn=None, models_path='./saved_models/'):
        self.epochs = num_epochs
        self.device = device
        self.lr = lr
        self.n_fft = n_fft
        self.loss_fn = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        self.models_path = models_path
        print('running on ' + str(self.device))

    def train(self, model, trainLoader):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if self.loss_fn is None:
            loss_fn = nn.MSELoss(reduction='none')
        else:
            loss_fn = self.loss_fn
        start_time = time.time()

        # if not isinstance(model, nn.DataParallel) and torch.cuda.device_count() > 3:
        #     model = nn.DataParallel(model)
        # model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)  # , weight_decay=1e-5)

        print("start training...")
        history = []
        train_losses = []
        mini_batch_size = 100
        print(datetime.datetime.now())
        for epoch in range(self.epochs):
            print('---------------')
            trainLoss = 0
            samples = 0
            model.train()
            running_loss = 0.0
            for i, batch in enumerate(trainLoader):
                x = batch['noisy'].to(self.device)
                clean_signals = batch['clean'].to(self.device)
                # stfts = torch.stft(batch['noisy'], self.n_fft, hop_length=128)
                # stfts = stfts.to(self.device)
                reconstructed_signal = model(x)
                # plt.imshow(decoded[0].detach().cpu().numpy().squeeze())
                batch_size = len(batch['clean'])
                loss = loss_fn(clean_signals.unsqueeze(1), reconstructed_signal.unsqueeze(1))
                # print('loss = ', loss)

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trainLoss += loss.item() * batch_size
                samples += batch_size

                if i % mini_batch_size == mini_batch_size - 1:  # print every 30 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / mini_batch_size:.3f}')
                    running_loss = 0.0
                    fig = plt.figure()
                    orig_sig = batch['clean'][0].detach().cpu().numpy().squeeze()
                    estimated_sig = reconstructed_signal[0].detach().cpu().numpy().squeeze()
                    noisy_sig = x[0].detach().cpu().numpy().squeeze()
                    plt.plot(orig_sig, c='b', label='orig signal')
                    plt.plot(estimated_sig, c='r', label='estimated signal')
                    plt.plot(noisy_sig, c='g', label='noisy signal')
                    plt.legend()
                    plt.show()
            # if i % mini_batch_size*10 == mini_batch_size - 1:  # print every 30 mini-batches
            model_name = self.models_path + f'model_{epoch + 1}_{trainLoss / samples}.pth'
            # best_valAcc = (valAcc / samples)
            torch.save(model.state_dict(), model_name)
