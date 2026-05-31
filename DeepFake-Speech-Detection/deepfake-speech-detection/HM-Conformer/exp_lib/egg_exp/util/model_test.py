import torch
import torch.nn.functional as F
import numpy as np

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import torch.distributed as dist

from .ddp_util import all_gather

def _flatten_labels(labels):
    """Flatten labels to ensure they're a 1D list of Python scalars.
    
    This handles cases where labels might be nested lists or contain
    numpy arrays/tensors after all_gather in DDP.
    """
    flattened = []
    for label in labels:
        if isinstance(label, (list, tuple)):
            # If it's a nested structure, recursively flatten
            flattened.extend(_flatten_labels(label))
        elif isinstance(label, np.ndarray):
            # Convert numpy array to list and flatten
            if label.ndim == 0:
                flattened.append(int(label.item()))
            else:
                flattened.extend([int(x) for x in label.flatten()])
        elif isinstance(label, torch.Tensor):
            # Convert tensor to Python scalar
            if label.numel() == 1:
                flattened.append(int(label.item()))
            else:
                # Multi-element tensor, flatten it
                flattened.extend([int(x.item()) for x in label.flatten()])
        else:
            # Already a scalar, convert to int
            flattened.append(int(label))
    return flattened

def calculate_EER(scores, labels):
    # Ensure labels are a flat list of integers
    labels = _flatten_labels(labels) if isinstance(labels, list) else labels
    scores = _flatten_labels(scores) if isinstance(scores, list) else scores
    
    # Convert to numpy arrays for sklearn
    labels = np.array(labels, dtype=np.int32)
    scores = np.array(scores, dtype=np.float64)
    
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def df_test(framework, loader, run_on_ddp=False, get_scores=False):
    '''Test deepfake detection and return EER 
    
    Param
        loader: DataLoader that returns (wav, label)
        get_scores: if True, returns the score and label used in EER calculation.
        
    Return
        eer(float)
        score(list(float))
        labels(list(int))
    '''
    framework.eval()

    labels = []
    scores = []
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, label in loader:
            if run_on_ddp:
                x = x.to(torch.float32).to(framework.device, non_blocking=True)
            else:
                x = x.to(torch.float32).to(framework.device)
            x = framework(x).to('cpu')
            
            for i in range(x.size(0)):
                if x.size(1) == 1:
                    scores.append(x[i, 0].item())
                else:
                    scores.append(x[i, 1].item())
                labels.append(label[i].item())
        
            pbar.update(1)
    
    if run_on_ddp:
        _synchronize()
        scores = all_gather(scores)
        labels = all_gather(labels)

    eer = calculate_EER(scores, labels)
    
    if get_scores:
        return eer, scores, labels
    else:
        return eer
    
def df_test_embd(framework, loader, run_on_ddp=False, get_scores=False):
    '''Test deepfake detection and return EER 
    
    Param
        loader: DataLoader that returns (wav, label)
        get_scores: if True, returns the score and label used in EER calculation.
        
    Return
        eer(float)
        score(list(float))
        labels(list(int))
    '''
    framework.eval()

    labels = [[],[],[],[],[]]
    scores = [[],[],[],[],[]]
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, label in loader:
            if run_on_ddp:
                x = x.to(torch.float32).to(framework.device, non_blocking=True)
            else:
                x = x.to(torch.float32).to(framework.device)
            xs = framework(x, all_loss=True)
            for j in range(5):
                x = xs[j].to('cpu')

                for i in range(x.size(0)):
                    if x.size(1) == 1:
                        scores[j].append(x[i, 0].item())
                    else:
                        scores[j].append(x[i, 1].item())
                    labels[j].append(label[i].item())
            
            pbar.update(1)
    
    if run_on_ddp:
        _synchronize()
        for j in range(5):
            scores[j] = all_gather(scores[j])
            labels[j] = all_gather(labels[j])
        print('s0',len(scores[0]),'    s1',len(scores[1]))

    EER = []
    for j in range(5):
        eer = calculate_EER(scores[j], labels[j])
        EER.append(eer)
        
    if get_scores:
        return EER, scores, labels
    else:
        return EER

def sv_enrollment(framework, loader):
    '''
    '''
    framework.eval()
    
    keys = []
    embeddings_full = []
    embeddings_seg = []

    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x_full, x_seg, key in loader:
            x_full = x_full.to(torch.float32).to(framework.device, non_blocking=True)
            x_seg = x_seg.to(torch.float32).to(framework.device, non_blocking=True).view(-1, x_seg.size(-1)) 
            
            x_full = framework(x_full).to('cpu')
            x_seg = framework(x_seg).to('cpu')
            
            keys.append(key[0])
            embeddings_full.append(x_full)
            embeddings_seg.append(x_seg)

            pbar.update(1)

    _synchronize()

    # gather
    keys = all_gather(keys)
    embeddings_full = all_gather(embeddings_full)
    embeddings_seg = all_gather(embeddings_seg)

    full_dict = {}
    seg_dict = {}
    for i in range(len(keys)):
        full_dict[keys[i]] = embeddings_full[i]
        seg_dict[keys[i]] = embeddings_seg[i]
            
    return full_dict, seg_dict
    
def sv_test(trials, single_embedding=None, multi_embedding=None):
    '''Calculate EER for test speaker verification performance.
    
    Param
        trials(list): list of SV_Trial (it contains key1, key2, label) 
        single_embedding(dict): embedding dict extracted from single utterance
        multi_embedding(dict): embedding dict extracted from multi utterance, such as TTA
    
    Return
        eer(float)
    '''
    labels = []
    cos_sims_full = [[], []]
    cos_sims_seg = [[], []]

    for item in trials:
        if single_embedding is not None:
            cos_sims_full[0].append(single_embedding[item.key1])
            cos_sims_full[1].append(single_embedding[item.key2])

        if multi_embedding is not None:
            cos_sims_seg[0].append(multi_embedding[item.key1])
            cos_sims_seg[1].append(multi_embedding[item.key2])

        labels.append(item.label)

    # cosine_similarity - full
    count = 0
    cos_sims = 0
    if single_embedding is not None:
        buffer1 = torch.cat(cos_sims_full[0], dim=0)
        buffer2 = torch.cat(cos_sims_full[1], dim=0)
        cos_sims_full = F.cosine_similarity(buffer1, buffer2)
        cos_sims = cos_sims + cos_sims_full
        count += 1

    # cosine_similarity - seg
    if multi_embedding is not None:
        batch = len(labels)
        num_seg = cos_sims_seg[0][0].size(0)
        buffer1 = torch.stack(cos_sims_seg[0], dim=0).view(batch, num_seg, -1)
        buffer2 = torch.stack(cos_sims_seg[1], dim=0).view(batch, num_seg, -1)
        buffer1 = buffer1.repeat(1, num_seg, 1).view(batch * num_seg * num_seg, -1)
        buffer2 = buffer2.repeat(1, 1, num_seg).view(batch * num_seg * num_seg, -1)
        cos_sims_seg = F.cosine_similarity(buffer1, buffer2)
        cos_sims_seg = cos_sims_seg.view(batch, num_seg * num_seg)
        cos_sims_seg = cos_sims_seg.mean(dim=1)
        cos_sims = cos_sims + cos_sims_seg
        count += 1

    cos_sims = (cos_sims_full + cos_sims_seg) / count
    eer = calculate_EER(cos_sims, labels)
    
    return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()