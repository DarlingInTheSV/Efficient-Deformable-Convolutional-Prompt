import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def cross_entropy_with_logits_(logits, target_probs):
    log_softmax_logits = F.log_softmax(logits, dim=1)
    return -(target_probs * log_softmax_logits).sum(dim=1).mean()


def cross_entropy_with_logits(logits, target_probs):
    logits = torch.sigmoid(logits)
    target_probs = torch.sigmoid(target_probs)
    loss1 = cross_entropy_with_logits_(logits, target_probs)
    loss2 = cross_entropy_with_logits_(target_probs, logits)
    return loss1 + loss2

def compute_joint_histogram(img1, img2, bins, min_val, max_val):
    """
    Compute the joint histogram for two images.
    """
    b = bins
    img1 = img1.view(-1)
    img2 = img2.view(-1)

    # Scale to range [0, b-1]
    img1_scaled = ((img1 - min_val) / (max_val - min_val) * (b - 1)).long()
    img2_scaled = ((img2 - min_val) / (max_val - min_val) * (b - 1)).long()

    joint_histogram = torch.zeros(b, b, dtype=torch.float32).to(img1.device)
    for i in range(img1.size(0)):
        joint_histogram[img1_scaled[i], img2_scaled[i]] += 1

    return joint_histogram

def mutual_information_loss(img1, img2, bins=20):
    """
    Compute mutual information loss between two images.
    """
    min_val = min(img1.min(), img2.min())
    max_val = max(img1.max(), img2.max())

    joint_hist = compute_joint_histogram(img1, img2, bins, min_val, max_val)

    # Convert joint histogram to joint probability distribution
    joint_prob = joint_hist / torch.sum(joint_hist)

    # Marginal probabilities
    p_x = torch.sum(joint_prob, dim=1)
    p_y = torch.sum(joint_prob, dim=0)

    # Calculate mutual information
    p_x_p_y = p_x.view(-1, 1) * p_y.view(1, -1)

    # Use mask to avoid log(0)
    non_zero_joint_prob = joint_prob > 0

    # Avoid division by zero
    safe_joint_prob = torch.where(non_zero_joint_prob, joint_prob, torch.tensor(1.0, device=joint_prob.device))
    safe_p_x_p_y = torch.where(non_zero_joint_prob, p_x_p_y, torch.tensor(1.0, device=p_x_p_y.device))

    mi = torch.sum(safe_joint_prob * torch.log(safe_joint_prob / safe_p_x_p_y))

    return -mi


def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def ssim_loss(img1, img2):
    return 1-ssim(img1, img2)



