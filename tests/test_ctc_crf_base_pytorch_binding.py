import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
import sys
sys.path.append('../../src/ctc_crf')
import ctc_crf_base
import numpy as np

TARGET_GPUS = [0]
gpus = torch.IntTensor(TARGET_GPUS)

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, "shouldn't require grads"

class _CTC_CRF(Function):
    @staticmethod
    def forward(ctx, logits, labels, input_lengths, label_lengths, lamb=0.1, size_average=False):
        logits = logits.contiguous() # n,t,d

        batch_size = logits.size(0)
        costs_alpha_den = torch.zeros(logits.size(0)).type_as(logits)
        costs_beta_den = torch.zeros(logits.size(0)).type_as(logits)

        grad_den = torch.zeros(logits.size()).type_as(logits)

        costs_ctc = torch.zeros(logits.size(0))
        act = torch.transpose(logits, 0, 1).contiguous() # t,n,d
        grad_ctc= torch.zeros(act.size()).type_as(logits)

        # print("act size:", act.size())
        # print("logits size:", logits.size())
        # print("batch size:", logits.size(0))
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
        # print("logits:", logits)
        # print("label_lengths:", labels)
        # print("input_lengths:", input_lengths)
        # print("label_lengths:", label_lengths)

        # print('>> gpu_ctc')
        ctc_crf_base.gpu_ctc(act, grad_ctc, labels, label_lengths, input_lengths, logits.size(0), costs_ctc, 0)
        # print('>> gpu_den')
        ctc_crf_base.gpu_den(logits, grad_den, input_lengths.cuda(), costs_alpha_den, costs_beta_den)

        grad_ctc = torch.transpose(grad_ctc, 0, 1)
        costs_ctc = costs_ctc.to(logits.get_device())

        grad_all = grad_den - (1 + lamb) * grad_ctc # n,t,d
        costs_all = costs_alpha_den - (1 + lamb) * costs_ctc # n
        costs = torch.FloatTensor([costs_all.sum()]).to(logits.get_device())

        if size_average:
            grad_all = grad_all / batch_size
            costs = costs / batch_size

        ctx.grads = grad_all
        print('>>>>>>>>> costs:')
        print(costs)
        print('>>>>>>>>> grad_all:')
        print(grad_all)

        return costs

    @staticmethod
    def backward(ctx, grad_output): # n,t,d
        return ctx.grads * grad_output.to(ctx.grads.device), None, None, None, None, None, None


class CTC_CRF_LOSS(Module):
    def __init__(self, lamb = 0.1, size_average=True):
        super(CTC_CRF_LOSS, self).__init__()
        self.ctc_crf = _CTC_CRF.apply
        self.lamb = lamb
        self.size_average = size_average
    
    def forward(self, logits, labels, input_lengths, label_lengths):
        # print(labels.size())
        assert len(labels.size()) == 1
        _assert_no_grad(labels)
        _assert_no_grad(input_lengths)
        _assert_no_grad(label_lengths)
        return self.ctc_crf(logits, labels, input_lengths, label_lengths, self.lamb, self.size_average)

LM_PATH = '/work/pubrepo/CAT/egs/aishell/data/den_meta/den_lm.fst'
# LM_PATH = '/data/203_data/datalist/CHN_PHV2/lm_for_ctc_crf/den_lm_20200401_v1_rmsil_uniq.fst'
# LM_PATH = '/data/203_data/datalist/CHN_PHV2/lm_for_ctc_crf/den_lm_20200401_v1_uniq.fst'

class Model(nn.Module):
    def __init__(self, lamb):
        super(Model, self).__init__()
        # self.net = BLSTM(idim,  hdim, n_layers, dropout=dropout)
        # self.linear = nn.Linear(hdim*2, K)
        self.loss_fn = CTC_CRF_LOSS( lamb=lamb )

    def forward(self, logits, labels_padded, input_lengths, label_lengths):
        # rearrange by input_lengths
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim()==1, "input_lengths should have only 1 dim"
        logits = torch.index_select(logits, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [labels_padded[i, :x] for i,x in enumerate(label_lengths)]
        labels = torch.cat(label_list)
        # netout, _ = self.net(logits, input_lengths)
        # netout = self.linear(netout)
        netout = F.log_softmax(logits, dim=2)
        loss = self.loss_fn(netout, labels, input_lengths, label_lengths)
        return loss


if __name__ == '__main__':
    device = torch.device("cuda:0")

    ctc_crf_base.init_env(LM_PATH, gpus)

    # Softmax logits for the following inputs:
    logits = np.array([
        [0.1, 0.6, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.1, 0.1]
    ], dtype=np.float32)

    # dimensions should be t, n, p: (t timesteps, n minibatches,
    # p prob of each alphabet). This is one instance, so expand
    # dimensions in the middle
    logits = np.expand_dims(logits, 0)
    labels = np.asarray([[1, 2]], dtype=np.int32)
    input_lengths = np.asarray([2], dtype=np.int32)
    label_lengths = np.asarray([2], dtype=np.int32)

    # print(logits.shape)

    model = Model(0.1)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)

    # self.data_batch.append([torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])
    loss = model(
        torch.FloatTensor(logits), 
        torch.IntTensor(labels), 
        torch.IntTensor(input_lengths), 
        torch.IntTensor(label_lengths))
    print(loss)
    # loss.backward(loss.new_ones(len(TARGET_GPUS)))
    # print(x)
    ctc_crf_base.release_env(gpus)
