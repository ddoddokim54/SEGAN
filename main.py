import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

import librosa
import librosa.display

from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from model import Generator, Discriminator
from utils import AudioDataset, emphasis

# Loss 2. Spectral Loss
def SpectralLoss(x, target): 
    '''
    output & target : tensor (Amplitude - time)
    outputSpec : Spectrogram of output tensor
    targetSpec : Spectrogram of target tensor
    ''' 
    outputSpec = torchaudio.transforms.Spectrogram(
            power=None, window_fn=lambda x: torch.hann_window(400, device='cuda'))(x)
    targetSpec = torchaudio.transforms.Spectrogram(
            power=None, window_fn=lambda x: torch.hann_window(400, device='cuda'))(target)
    gap = outputSpec - targetSpec
    w, h, m, n = targetSpec.size()
    loss = torch.sum(gap.pow(2))/(2*w*h)

    return loss



# Loss 3. Frequency Cropped Loss
def FreqCropLoss(x, target) :
    '''
    x, target : spectrogrammed data with librosa.stft & amplitude to db.
    beta : weight array
        (length : n+1; beta[0]=0 , for convenience of index)
    freq : partition boundary frequency array
        (length : n+1; freq[0]=0 & freq[n]=sampling rate, same reason with above)

    Default sampling rate of Librosa is 22050Hz -> so i set max Hz value as 11025Hz.
    '''
    beta = [0, 0.3, 0.1, 0.2, 0.1, 0.3]
    freq = np.array([0, 400, 600, 1500, 3000, 11025])/11025
    partitions = len(freq)

    outputSpec = torchaudio.transforms.Spectrogram(
            power=None, window_fn=lambda x: torch.hann_window(400, device='cuda'))(x)
    targetSpec = torchaudio.transforms.Spectrogram(
            power=None, window_fn=lambda x: torch.hann_window(400, device='cuda'))(target)
    gap = outputSpec - targetSpec
    w, h, m, n = targetSpec.size()
    # w : time scale, h : freq scale

    loss = 0.
    for k in range(1, partitions-1) :
        l = 0.
        start  = int(freq[k-1]*h)
        finish = int(freq[k]*h)

        for i in range(start, finish) :
            l += torch.sum(gap.pow(2), dim=0)[i]/2
        loss += torch.sum(l)*beta[k]/(finish-start)
    return loss/w


# https://quokkas.tistory.com/37
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Validation loss  ({self.val_loss_min:.4f} --> {val_loss:.4f}).')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.val_loss_min = val_loss

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='test')
    validation_dataset = AudioDataset(data_type='validation')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=4)
                                        
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)
    
    #initializing early stopping
    early_stopping = EarlyStopping(patience = 20, verbose = True)
    
    for epoch in range(NUM_EPOCHS):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:

            # latent vector - normal distribution
            z = nn.init.normal(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            noisy_loss.backward()

            # d_loss = clean_loss + noisy_loss
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            # original loss
            # g_loss = g_loss_ + g_cond_loss
            
            # freqcroploss
            g_loss = g_loss_ + g_cond_loss + 100*FreqCropLoss(generated_outputs, train_clean)

            # spectralloss
            # g_loss = g_loss_ + g_cond_loss + 0.0000005*SpectralLoss(generated_outputs, train_clean)

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()

            train_bar.set_description(
                'Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
                    .format(epoch + 1, clean_loss.data, noisy_loss.data, g_loss.data, g_cond_loss.data))
        
        # for valid loss and early Stopping
        valid_loss = []
        
        validation_bar = tqdm(validation_data_loader, desc = 'Data Validation')
        for validation_batch, validation_clean, validation_noisy in validation_bar:
            
            z = nn.init.normal(torch.Tensor(validation_noisy.size(0), 1024, 8))
            if torch.cuda.is_available():
                validation_noisy, z = validation_noisy.cuda(), z.cuda()
                validation_clean = validation_clean.cuda()
            validation_noisy, z = Variable(validation_noisy), Variable(z)      
            generated_outputs = generator(validation_noisy, z)
            
            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(validation_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss
            valid_loss.append(g_loss.cpu().detach().numpy())
        
        valid_loss = np.mean(np.array(valid_loss))
        validation_bar.set_description(
            'Epoch {} : validation_loss {:.4f}'
                .format(epoch+1,valid_loss))
        print(valid_loss)
        early_stopping(valid_loss)
        
        if early_stopping.early_stop:
            print("")
            print("Early Stopping")
            g_path = os.path.join('epochs', 'Earlystop_generator-{}.pkl'.format(epoch + 1))
            d_path = os.path.join('epochs', 'Earlystop_discriminator-{}.pkl'.format(epoch + 1))
            
            break
            
        
        # TEST model
        test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        for test_file_names, test_noisy in test_bar:
            z = nn.init.normal(torch.Tensor(test_noisy.size(0), 1024, 8))
            if torch.cuda.is_available():
                test_noisy, z = test_noisy.cuda(), z.cuda()
            test_noisy, z = Variable(test_noisy), Variable(z)
            fake_speech = generator(test_noisy, z).data.cpu().numpy()  # convert to numpy array
            fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)

            for idx in range(fake_speech.shape[0]):
                generated_sample = fake_speech[idx]
                file_name = os.path.join('results',
                                         '{}_e{}.wav'.format(test_file_names[idx].replace('.npy', ''), epoch + 1))
                wavfile.write(file_name, sample_rate, generated_sample.T)

        # save the model parameters for each epoch
        g_path = os.path.join('epochs', 'generator-{}.pkl'.format(epoch + 1))
        d_path = os.path.join('epochs', 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
