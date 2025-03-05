
import torch
import torch.nn as nn
import math

import logging
logger = logging.getLogger('FLD')
import torch.nn.functional as F

class HGNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super(HGNN, self).__init__()
        #
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)
    def forward(self, x, G):
        residual = x

        x = self.conv(x)

        x = G.matmul(x)
        x = F.relu(self.bn(x.permute(0,2,1).contiguous())).permute(0,2,1).contiguous() + residual
        return x
class DS_HGNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super(DS_HGNN, self).__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x, residual):
        residual = residual

        # 这里的 matmul 是 PyTorch 中用于矩阵乘法的函数。

        # print('x.shape', x.shape)
        x = F.relu(self.bn(x.permute(0,2,1).contiguous())).permute(0,2,1).contiguous() + residual
        return x

# 超图神经网络层
class DS_HGNN_layer(nn.Module):


    def __init__(self, in_ch, node = None, K_neigs=None, kernel_size=5, stride=2):
        super(DS_HGNN_layer, self).__init__()
        self.HGNN = DS_HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs

        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)
        self.step = 20
        print('in_ch', in_ch)
        self.bn1 = nn.BatchNorm1d(node*node)
        self.bn2 = nn.BatchNorm1d(node*node)
        self.bn = 1

        self.theta_vertex = nn.Linear(in_ch, in_ch, bias=True)
        self.theta_hyperedge = nn.Linear(in_ch, in_ch, bias=True)
        self.layer_num = 40
        self.act =nn.ReLU(inplace=True)
        self.alpha_v = 0.05
        self.alpha_e = 0.9

        self.drop = nn.Dropout(0.15)

    def forward(self, X):


        B, N, C = X.shape


        r = X
        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(X, k=self.K_neigs[0])




        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)


        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.
        Dv = Dv * alpha
        max_k = int(Dv.max())

        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(X, k=max_k - 1)


        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(X.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(X.device)

        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()
        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)
        local_H = self.local_H.repeat(B,1,1).to(new_H.device)

        _H = torch.cat([new_H,local_H],dim=2)




        DV = torch.sum(_H, dim=2)

        DE = torch.sum(_H, dim=1)

        HT = _H.transpose(1, 2)


        E = torch.diag_embed(torch.pow(DE, -1)) @ HT @ X

        W_ev = torch.diag_embed((torch.pow(DE, -1))) @ (HT)
        W_ve = torch.diag_embed((torch.pow(DV, -1))) @ (_H)

        for i in range(self.layer_num):

            if i % self.step == 0:
                X = X + self.act(self.theta_vertex(X))
                E = E + self.act(self.theta_hyperedge(E))

                if self.bn is not None:



                    X = nn.BatchNorm1d(X.shape[-2]).to(X.device)(X)
                    E = nn.BatchNorm1d(E.shape[-2]).to(E.device)(E)
            X = self.drop(X)

            newX = X - self.alpha_v * (X - W_ve @ (E))

            newE = E - self.alpha_e * (E - W_ev @ (X))

            X = newX

            E = newE


        x = self.HGNN(X, r)

        return x



    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape


        # the weight of the hyperedge
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)

        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedg
        DE = torch.sum(H, dim=1)


        invDE = torch.diag_embed((torch.pow(DE, -1)))

        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)



        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G

    @torch.no_grad()
    def _generate_DG_from_H_b(self, H, X, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape

        # the weight of the hyperedge
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)

        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedg
        DE = torch.sum(H, dim=1)

        HT = H.transpose(1, 2)

        ##
        E = torch.diag_embed((torch.pow(DE, -1))).mm(HT).mm(X)
        W_ev = torch.diag_embed((torch.pow(DE, -1))).mm(HT)
        W_ve = torch.diag_embed((torch.pow(DV, -1))).mm(H)

        for i in range(self.layer_num):

            if i % self.step == 0:
                X = X + self.act(self.theta_vertex(X))
                E = E + self.act(self.theta_hyperedge(E))
                if self.bn is not None:
                    X = self.bn(X)
                    E = self.bn(E)
            X = self.drop(X)
            newX = X - self.alpha_v * (X - W_ve.mm(E))
            newE = E - self.alpha_e * (E - W_ev.mm(X))

            X = newX

            E = newE

        X = self.classifier(X)
        return X






    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)

        avg_dists = ori_dists.mean(-1, keepdim=True)

        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            # Chai's weight function
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):

        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]
        print('inp.shape', inp.shape)

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), dilation=1, stride=stride).squeeze(
            0).transpose(0, 1).long()
        print('kernal_size', kernel_size)
        print('inp_unf.shape', inp_unf.shape)



        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()
        print('matrix.shape', matrix.shape)

        H_local = torch.zeros((size * size, edge))


        H_local[inp_unf, matrix] = 1.
        print('H_local.shape', H_local.shape)

        return H_local


class DS_HGNN_layer_updata(nn.Module):
    """
        Writen by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, node = None, K_neigs=None, kernel_size=5, stride=2):
        super(DS_HGNN_layer_updata, self).__init__()
        self.HGNN = DS_HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs
        # self.layer = 4
        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)
        self.step = 4
        print('in_ch', in_ch)
        self.bn1 = nn.BatchNorm1d(node*node)
        self.bn2 = nn.BatchNorm1d(node*node)
        self.bn = 1

        self.theta_vertex = nn.Linear(in_ch, in_ch, bias=True)
        self.theta_hyperedge = nn.Linear(in_ch, in_ch, bias=True)
        self.layer_num = 8
        self.act =nn.ReLU(inplace=True)
        self.alpha_v = 0.05
        self.alpha_e = 0.9

        self.drop = nn.Dropout(0.15)

    def forward(self, X):





        r = X



        for i in range(self.layer_num):
            _H, HT, E, W_ev, W_ve = self._generate_DG_from_H_b(X)

            if i % self.step == 0:
                X = X + self.act(self.theta_vertex(X))
                E = E + self.act(self.theta_hyperedge(E))

                if self.bn is not None:



                    X = nn.BatchNorm1d(X.shape[-2]).to(X.device)(X)
                    E = nn.BatchNorm1d(E.shape[-2]).to(E.device)(E)
            X = self.drop(X)

            newX = X - self.alpha_v * (X - W_ve @ (E))

            newE = E - self.alpha_e * (E - W_ev @ (X))

            X = newX

            E = newE


        x = self.HGNN(X, r)

        return x



    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape


        # the weight of the hyperedge
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)

        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedg
        DE = torch.sum(H, dim=1)


        invDE = torch.diag_embed((torch.pow(DE, -1)))

        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)



        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G

    @torch.no_grad()
    def _generate_DG_from_H_b(self, X):
        B, N, C = X.shape
        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(X, k=self.K_neigs[0])


        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)


        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.
        Dv = Dv * alpha
        max_k = int(Dv.max())

        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(X, k=max_k - 1)

        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(X.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(X.device)

        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()
        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)
        local_H = self.local_H.repeat(B, 1, 1).to(new_H.device)

        _H = torch.cat([new_H, local_H], dim=2)


        DV = torch.sum(_H, dim=2)

        DE = torch.sum(_H, dim=1)

        HT = _H.transpose(1, 2)

        E = torch.diag_embed(torch.pow(DE, -1)) @ HT @ X

        W_ev = torch.diag_embed((torch.pow(DE, -1))) @ (HT)
        W_ve = torch.diag_embed((torch.pow(DV, -1))) @ (_H)
        return _H, HT, E, W_ev, W_ve








    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)

        avg_dists = ori_dists.mean(-1, keepdim=True)

        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            # Chai's weight function
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):

        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]
        print('inp.shape', inp.shape)

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()
        print('kernal_size', kernel_size)
        print('inp_unf.shape', inp_unf.shape)


        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()
        print('matrix.shape', matrix.shape)

        H_local = torch.zeros((size * size, edge))


        H_local[inp_unf, matrix] = 1.
        print('H_local.shape', H_local.shape)

        return H_local



class HGNN_layer(nn.Module):
    """
        Writen by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, node = None, K_neigs=None, kernel_size=5, stride=2):
        super(HGNN_layer, self).__init__()
        self.HGNN = HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs

        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)

    def forward(self, x):


        B, N, C = x.shape

        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(x, k=self.K_neigs[0])


        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)

        Dv = torch.sum(H, dim=2, keepdim=True)

        alpha = 1.
        Dv = Dv * alpha

        max_k = int(Dv.max())
        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(x, k=max_k - 1)
        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(x.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(x.device)
        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()

        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)

        local_H = self.local_H.repeat(B,1,1).to(new_H.device)

        _H = torch.cat([new_H,local_H],dim=2)
        _G = self._generate_G_from_H_b(_H)

        x = self.HGNN(x, _G)

        return x



    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape


        # the weight of the hyperedge
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)

        DV = torch.sum(H, dim=2)
        # the degree of the hyperedge
        DE = torch.sum(H, dim=1)


        invDE = torch.diag_embed((torch.pow(DE, -1)))

        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))

        W = torch.diag_embed(W)

        HT = H.transpose(1, 2)



        if variable_weight:

            DV2_H = DV2 @ H

            invDE_HT_DV2 = invDE @ HT @ DV2

            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G


    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)

        avg_dists = ori_dists.mean(-1, keepdim=True)

        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):

        B, N, K = top_dists.shape

        weights = self.weights_function(top_dists, avg_dists, prob)    # shape: (batch_size, num_points, k)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]

        incidence_matrix[batch_indices, pixel_indices, inds] = weights
        # (batch_size, num_points, num_points)
        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):

        if prob:

            topk_dists_sq = topk_dists.pow(2)

            normalized_topk_dists_sq = topk_dists_sq / avg_dists

            weights = torch.exp(-normalized_topk_dists_sq)
        else:

            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    def local_kernel(self, size, kernel_size=3, stride=1):

        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()

        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()

        H_local = torch.zeros((size * size, edge))


        H_local[inp_unf, matrix] = 1.

        return H_local



class HyperNet(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(HyperNet, self).__init__()

        self.node = node

        self.channel = channel

        self.K_neigs = K_neigs

        self.kernel_size = kernel_size
        self.stride = stride

        self.HGNN_layer = DS_HGNN_layer(self.channel, node = self.node, kernel_size=self.kernel_size, stride=self.stride, K_neigs=self.K_neigs)

    def forward(self, x):

        b,c,w,h = x.shape

        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        x = self.HGNN_layer(x)

        x = x.permute(0,2,1).contiguous().view(b,c,w,h)

        return x

class HyperNet_updata(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(HyperNet_updata, self).__init__()

        self.node = node

        self.channel = channel

        self.K_neigs = K_neigs

        self.kernel_size = kernel_size
        self.stride = stride

        self.HGNN_layer = DS_HGNN_layer_updata(self.channel, node = self.node, kernel_size=self.kernel_size, stride=self.stride, K_neigs=self.K_neigs)

    def forward(self, x):

        b,c,w,h = x.shape

        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        x = self.HGNN_layer(x)

        x = x.permute(0,2,1).contiguous().view(b,c,w,h)

        return x



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None, node=None):
        super(BasicBlock, self).__init__()
        ##
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockHGNN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None, node=10):
        super(BasicBlockHGNN, self).__init__()
        ##
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.node = node

        self.HGNN_layer = HyperNet(channel=inplanes, node=self.node, kernel_size=3, stride=1, K_neigs=[1])

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.HGNN_layer(residual)
            residual = self.downsample(residual)
        else:
            residual = self.HGNN_layer(residual)
            pass


        out += residual
        out = self.relu(out)


        return out

class BasicBlockHGNN_pure(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None, node=10):
        super(BasicBlockHGNN_pure, self).__init__()
        ##
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.node = node
        if stride == 2:
            self.n = self.node // 2
        else:
            self.n = self.node
        print(node)

        self.HGNN_layer2 = HyperNet(channel=planes, node=self.n, kernel_size=3, stride=1, K_neigs=[1])

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.HGNN_layer2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.downsample(residual)
        else:

            pass


        out += residual
        out = self.relu(out)


        return out
class BasicBlockHGNN_pure_updata(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None, node=10):
        super(BasicBlockHGNN_pure_updata, self).__init__()
        ##
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.node = node
        if stride == 2:
            self.n = self.node // 2
        else:
            self.n = self.node
        print(node)

        self.HGNN_layer1 = HyperNet_updata(channel=inplanes, node=self.node, kernel_size=3, stride=1, K_neigs=[1])
        self.HGNN_layer2 = HyperNet_updata(channel=planes, node=self.n, kernel_size=3, stride=1, K_neigs=[1])

    def forward(self, x):
        residual = x

        out = self.HGNN_layer1(x)
        out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.HGNN_layer2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.downsample(residual)
        else:
            # residual = self.HGNN_layer(residual)
            pass


        out += residual
        out = self.relu(out)


        return out


class BasicBlockHGNN_pure1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None, node=10):
        super(BasicBlockHGNN_pure1, self).__init__()
        ##
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.node = node

        self.HGNN_layer1 = HyperNet(channel=inplanes, node=self.node, kernel_size=3, stride=1, K_neigs=[1])


    def forward(self, x):
        residual = x

        out = self.HGNN_layer1(x)
        out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.downsample(residual)
        else:

            pass


        out += residual
        out = self.relu(out)


        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilate, padding=dilate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_feats, is_color=True):
        self.inplanes = num_feats[0]
        super(ResNet, self).__init__()


        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_feats[0], layers[0], stride=1, dilate=1)

        self.layer2 = self._make_layer(block, num_feats[1], layers[1], stride=2, dilate=1)

        self.layer3 = self._make_layer(block, num_feats[2], layers[2], stride=2, dilate=1)

        self.layer4 = self._make_layer(block, num_feats[3], layers[3], stride=2, dilate=1)
        #
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(5120 * block.expansion, 1)

        self.num_out_feats = [num_feat*block.expansion for num_feat in num_feats]
        print('num_feats', self.num_out_feats)
        self.downsample_ratio = 32

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        downsample = None
        # self.inplanes = 64
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilate, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        out1 = self.layer1(x)

        out2 = self.layer2(out1)

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)

        out5 = self.avgpool(out4)
        out6 = None


        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4, 'out5':out5, 'out6':out6}
        return x_dict


class ResNet_HGNN(nn.Module):

    def __init__(self, block1, block2, layers, num_feats, is_color=True):
        self.inplanes = num_feats[0]
        super(ResNet_HGNN, self).__init__()
        # 224 112 56 28 14 7
        self.node = [56, 56, 28, 14]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block1, num_feats[0], layers[0], stride=1, dilate=1, node=self.node[0])
        self.layer2 = self._make_layer(block1, num_feats[1], layers[1], stride=2, dilate=1, node=self.node[1])
        self.layer3 = self._make_layer(block2, num_feats[2], layers[2], stride=2, dilate=1, node=self.node[2])
        self.layer4 = self._make_layer(block2, num_feats[3], layers[3], stride=2, dilate=1, node=self.node[3])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block1.expansion, 1000)

        self.num_out_feats = [num_feat*block1.expansion for num_feat in num_feats]
        self.downsample_ratio = 32

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    ## [64, 128, 256, 512]
    # [64, 128, 256, 512]
    def _make_layer(self, block, planes, blocks, stride=1, dilate=1, node=100):
        downsample = None
        # self.inplanes = 64
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilate, downsample, node=node))
        if stride == 2:
            node = int(node / 2)
        self.inplanes = planes * block.expansion


        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, dilate, node=node))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(1000*"*")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('1', x.shape)
        out1 = self.layer1(x)
        # print('2', out1.shape)
        out2 = self.layer2(out1)
        # print('3', out2.shape)
        out3 = self.layer3(out2)
        # print('4', out3.shape)
        out4 = self.layer4(out3)
        # print('5', out4.shape)
        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict
class ResNet_HGNN_V1(nn.Module):

    def __init__(self, block1, block2, layers, num_feats, is_color=True):
        self.inplanes = num_feats[0]
        super(ResNet_HGNN_V1, self).__init__()
        # 224 112 56 28 14 7
        self.node = [56, 56, 28, 14]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block2, num_feats[0], layers[0], stride=1, dilate=1, node=self.node[0])
        self.layer2 = self._make_layer(block2, num_feats[1], layers[1], stride=2, dilate=1, node=self.node[1])
        self.layer3 = self._make_layer(block1, num_feats[2], layers[2], stride=2, dilate=1, node=self.node[2])
        self.layer4 = self._make_layer(block1, num_feats[3], layers[3], stride=2, dilate=1, node=self.node[3])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block1.expansion, 1000)

        self.num_out_feats = [num_feat*block1.expansion for num_feat in num_feats]
        self.downsample_ratio = 32

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    ## [64, 128, 256, 512]
    # [64, 128, 256, 512]
    def _make_layer(self, block, planes, blocks, stride=1, dilate=1, node=100):
        downsample = None
        # self.inplanes = 64
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilate, downsample, node=node))
        if stride == 2:
            node = int(node / 2)
        self.inplanes = planes * block.expansion


        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, dilate, node=node))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(1000*"*")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('1', x.shape)
        out1 = self.layer1(x)
        # print('2', out1.shape)
        out2 = self.layer2(out1)
        # print('3', out2.shape)
        out3 = self.layer3(out2)
        # print('4', out3.shape)
        out4 = self.layer4(out3)
        # print('5', out4.shape)
        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict



class HGNN_V2(nn.Module):

    def __init__(self, block1, block2, layers, num_feats, is_color=True):
        self.inplanes = num_feats[0]
        super(HGNN_V2, self).__init__()
        # 224 112 56 28 14 7
        self.node = [56, 56, 28, 14]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block1, num_feats[0], layers[0], stride=1, dilate=1, node=self.node[0])
        self.layer2 = self._make_layer(block1, num_feats[1], layers[1], stride=2, dilate=1, node=self.node[1])
        self.layer3 = self._make_layer(block1, num_feats[2], layers[2], stride=2, dilate=1, node=self.node[2])
        self.layer4 = self._make_layer(block2, num_feats[3], layers[3], stride=2, dilate=1, node=self.node[3])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block1.expansion, 1000)

        self.num_out_feats = [num_feat*block1.expansion for num_feat in num_feats]
        self.downsample_ratio = 32

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1, node=100):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilate, downsample, node=node))
        if stride == 2:
            node = int(node / 2)
        self.inplanes = planes * block.expansion


        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, dilate, node=node))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        # print('2', out1.shape)
        out2 = self.layer2(out1)
        # print('3', out2.shape)
        out3 = self.layer3(out2)
        # print('4', out3.shape)
        out4 = self.layer4(out3)
        # print('5', out4.shape)
        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


from timm.models.layers import DropPath



class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        # 激活函数
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        shortcut = x
        # x = x.permute(0, 2, 1).unsqueeze(-1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        # x = x.squeeze(-1).permute(0, 2, 1)
        x = x + shortcut
        return x



class ResNet_HGNN_V3(nn.Module):

    def __init__(self, block1, block2, layers, num_feats, is_color=True):
        self.inplanes = num_feats[0]
        super(ResNet_HGNN_V3, self).__init__()

        self.node = [56, 56, 28, 14]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block1, num_feats[0], layers[0], stride=1, dilate=1, node=self.node[0])
        self.layer2 = self._make_layer(block1, num_feats[1], layers[1], stride=2, dilate=1, node=self.node[1])
        self.layer3 = self._make_layer(block1, num_feats[2], layers[2], stride=2, dilate=1, node=self.node[2])
        self.layer4 = self._make_layer(block2, num_feats[3], layers[3], stride=2, dilate=1, node=self.node[3], flag=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block1.expansion, 1000)

        self.num_out_feats = [num_feat*block1.expansion for num_feat in num_feats]
        self.downsample_ratio = 32

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    ## [64, 128, 256, 512]
    # [64, 128, 256, 512]
    def _make_layer(self, block, planes, blocks, stride=1, dilate=1, node=100, flag=None):
        downsample = None
        # self.inplanes = 64
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilate, downsample, node=node))

        if flag == 2:
            layers.append(FFN(planes, planes*4))
        if stride == 2:
            node = int(node / 2)
        self.inplanes = planes * block.expansion


        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, dilate, node=node))
            if flag == 2:
                layers.append(FFN(planes, planes*4))


        return nn.Sequential(*layers)

    def forward(self, x):
        # print(1000*"*")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('1', x.shape)
        out1 = self.layer1(x)
        # print('2', out1.shape)
        out2 = self.layer2(out1)
        # print('3', out2.shape)
        out3 = self.layer3(out2)
        # print('4', out3.shape)
        out4 = self.layer4(out3)
        # print('5', out4.shape)
        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict





def HFEbackbone(is_color, pretrained_path=None, receptive_keep=False):
    logger.debug("build HFEbackbone ......")
    return HGNN_V2(block1=BasicBlock, block2=BasicBlockHGNN_pure, layers=[3, 4, 6, 1],
                          num_feats=[64, 128, 256, 512],
                          is_color=is_color)





if __name__ == '__main__':
    resnet = HFEbackbone(is_color=True).cuda()

