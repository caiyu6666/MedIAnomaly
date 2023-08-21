import torch
import torch.nn as nn

# def l2_loss(net_in, net_out, keepdim=False, anomaly_score=False):
#     rec = net_out['x_hat']
#     loss = ((net_in - rec) ** 2)
#
#     if anomaly_score:
#         return torch.mean(loss, dim=[1, 2, 3])
#     else:
#         return loss if keepdim else loss.mean()
#
#
# def l1_loss(net_in, net_out, keepdim=False, anomaly_score=False):
#     x_hat = net_out['x_hat']
#     loss = torch.abs(net_in - x_hat)
#
#     if anomaly_score:
#         return torch.mean(loss, dim=[1, 2, 3])
#     else:
#         return loss if keepdim else loss.mean()


def ae_loss(net_in, net_out, anomaly_score=False, keepdim=False):
    x_hat = net_out['x_hat']
    loss = (net_in - x_hat) ** 2

    if anomaly_score:
        return loss if keepdim else torch.mean(loss, dim=[1, 2, 3])
    else:
        return loss.mean()


def ae_loss_grad(net_in, net_out, anomaly_score=False, keepdim=False):
    x_hat = net_out['x_hat']
    loss = (net_in - x_hat) ** 2

    if anomaly_score:
        grad = torch.abs(torch.autograd.grad(loss.mean(), net_in)[0])
        return grad if keepdim else torch.mean(grad, dim=[1, 2, 3])
    else:
        return loss.mean()


# ----------- AE-U Loss ----------- #
def aeu_loss(net_in, net_out, anomaly_score=False, keepdim=False):
    x_hat, log_var = net_out['x_hat'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2

    loss1 = torch.exp(-log_var) * recon_loss

    loss = loss1 + log_var

    if anomaly_score:
        return loss1 if keepdim else torch.mean(loss1, dim=[1, 2, 3])
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


def memae_loss(net_in, net_out, entropy_loss_weight=0.0002, anomaly_score=False, keepdim=False):
    x_hat, att = net_out['x_hat'], net_out['att']
    recon_loss = (net_in - x_hat) ** 2
    entro_loss = entropy_loss(att)
    loss = recon_loss.mean() + entropy_loss_weight * entro_loss

    if anomaly_score:
        return recon_loss if keepdim else torch.mean(recon_loss, dim=[1, 2, 3])
    else:
        return loss, recon_loss.mean().item(), entro_loss.item()


# ----------- VAE Loss ----------- #
def vae_loss(net_in, net_out, kl_weight=0.005, anomaly_score=False, keepdim=False):
    x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2
    # recon_loss = torch.abs(net_in - x_hat)
    kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recon_loss.mean() + kl_weight * kl_loss.mean()

    if anomaly_score:
        return recon_loss if keepdim else torch.mean(recon_loss, dim=[1, 2, 3])
    else:
        return loss, recon_loss.mean().item(), kl_loss.mean().item()


def vae_loss_grad_elbo(net_in, net_out, kl_weight=0.005, anomaly_score=False, keepdim=False):
    x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2
    kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recon_loss.mean() + kl_weight * kl_loss.mean()

    if anomaly_score:
        grad = torch.abs(torch.autograd.grad(loss, net_in)[0])
        return grad if keepdim else torch.mean(grad, dim=[1, 2, 3])
    else:
        return loss, recon_loss.mean().item(), kl_loss.mean().item()


def vae_loss_grad_rec(net_in, net_out, kl_weight=0.005, anomaly_score=False, keepdim=False):
    x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2
    kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recon_loss.mean() + kl_weight * kl_loss.mean()

    if anomaly_score:
        grad = torch.abs(torch.autograd.grad(recon_loss.mean(), net_in)[0])
        return grad if keepdim else torch.mean(grad, dim=[1, 2, 3])
    else:
        return loss, recon_loss.mean().item(), kl_loss.mean().item()


def vae_loss_grad_kl(net_in, net_out, kl_weight=0.005, anomaly_score=False, keepdim=False):
    x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2
    kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recon_loss.mean() + kl_weight * kl_loss.mean()

    if anomaly_score:
        grad = torch.abs(torch.autograd.grad(kl_loss.mean(), net_in)[0])
        return grad if keepdim else torch.mean(grad, dim=[1, 2, 3])
    else:
        return loss, recon_loss.mean().item(), kl_loss.mean().item()


def vae_loss_grad_combi(net_in, net_out, kl_weight=0.005, anomaly_score=False, keepdim=False):
    x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
    recon_loss = (net_in - x_hat) ** 2
    # recon_loss = torch.abs(net_in - x_hat)
    kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recon_loss.mean() + kl_weight * kl_loss.mean()

    if anomaly_score:
        kl_grad = torch.abs(torch.autograd.grad(kl_loss.mean(), net_in)[0])
        combi = recon_loss * kl_grad
        return combi if keepdim else torch.mean(combi, dim=[1, 2, 3])
    else:
        return loss, recon_loss.mean().item(), kl_loss.mean().item()


def ganomaly_loss(net_in, net_out, mode='g', w_adv=1, w_rec=50, w_enc=1, anomaly_score=False):
    assert mode in ['g', 'd']  # compute loss of generator or discriminator

    if anomaly_score:
        z, z_hat = net_out['z'], net_out['z_hat']
        return torch.mean((z - z_hat) ** 2, dim=1)
    else:
        if mode == 'g':
            x_hat, z, z_hat, feat_real, feat_fake = \
                net_out['x_hat'], net_out['z'], net_out['z_hat'], net_out['feat_real'], net_out['feat_fake']
            loss_adv = torch.mean((feat_real - feat_fake) ** 2)
            loss_rec = torch.mean((net_in - x_hat) ** 2)
            loss_enc = torch.mean((z - z_hat) ** 2)

            loss_g = w_adv * loss_adv + w_rec * loss_rec + w_enc * loss_enc
            return loss_g, loss_adv.item(), loss_rec.item(), loss_enc.item()
        else:
            l_bce = nn.BCELoss()
            pred_real, pred_fake_detach = net_out['pred_real'], net_out['pred_fake_detach']
            real_label = torch.ones(size=(pred_real.shape[0],), dtype=torch.float32).cuda()
            fake_label = torch.zeros(size=(pred_fake_detach.shape[0],), dtype=torch.float32).cuda()
            loss_d = (l_bce(pred_real, real_label) + l_bce(pred_fake_detach, fake_label)) * 0.5
            return loss_d


def constrained_ae_loss(net_in, net_out, anomaly_score=False, keepdim=False):
    x_hat = net_out['x_hat']
    z = net_out['z']
    loss_x = (net_in - x_hat) ** 2

    if anomaly_score:
        return loss_x if keepdim else torch.mean(loss_x, dim=[1, 2, 3])
    else:
        z_rec = net_out['z_rec']
        loss_z = (z - z_rec) ** 2
        loss = loss_x.mean() + loss_z.mean()
        return loss.mean(), loss_x.mean().item(), loss_z.mean().item()


def fanogan_loss(net_in, net_out, mode='g', anomaly_score=False, keepdim=False):
    # TODO
    pass
