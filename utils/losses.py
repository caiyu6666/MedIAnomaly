import torch


def l2_loss(net_in, net_out, keepdim=False):
    rec = net_out['x_hat']
    loss = ((net_in - rec) ** 2)
    return loss if keepdim else loss.mean()


def l1_loss(net_in, net_out, keepdim=False):
    x_hat = net_out['x_hat']
    loss = torch.abs(net_in - x_hat)
    return loss if keepdim else loss.mean()


# ----------- AE-U Loss ----------- #
def aeu_loss(net_in, net_out, keepdim=False):
    x_hat, log_var = net_out['x_hat'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2

    loss1 = torch.exp(-log_var) * recon_loss

    loss = loss1 + log_var

    if keepdim:
        return loss1
    else:
        return loss.mean(), recon_loss.mean().item(), log_var.mean().item()


# ----------- MemAE Loss ----------- #
def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxDxHxW --> NxDxHxW x C
    if l == 2:
        x = input  # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxDxHxW x C --> (NxDxHxW) x C
    x = x.view(-1, s[1])
    return x


def entropy_loss(x, eps=1e-12):
    x = feature_map_permute(x)
    b = x * torch.log(x + eps)
    b = -1. * b.sum(dim=1)
    return b.mean()


def memae_loss(net_in, net_out, entropy_loss_weight=0.0002, keepdim=False):
    x_hat, att = net_out['x_hat'], net_out['att']
    recon_loss = (net_in - x_hat) ** 2
    entro_loss = entropy_loss(att)
    loss = recon_loss.mean() + entropy_loss_weight * entro_loss

    if keepdim:
        return recon_loss
    else:
        return loss, recon_loss.mean().item(), entro_loss.item()
