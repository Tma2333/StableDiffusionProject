import torch.nn.functional as F
import torch.nn as nn
import torch

class ContentLoss(nn.Module):

    def __init__(self, target,weight):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.target = target.detach()

    def forward(self, inputs):
        length = min(len(self.target), len(inputs))
        self.loss = self.weight*F.mse_loss(inputs[:length], self.target[:length])#/length
        return inputs



def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix

    G = G.div(torch.norm(G))
    return G

class StyleLoss(nn.Module):

    def __init__(self, reference_imgs, weight):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.Gs_ref = [gram_matrix(inp.unsqueeze(0)).detach() for inp in reference_imgs]

    def forward(self, inputs):
        Gs = [gram_matrix(inp.unsqueeze(0)) for inp in inputs]
        self.loss = 0
        for i in range(len(Gs)):
            for j in range(0, len(self.Gs_ref)):
                self.loss = torch.add(self.loss, F.mse_loss(Gs[i], self.Gs_ref[j]))
        self.loss = self.weight*self.loss.div(len(Gs)*len(self.Gs_ref))
        return inputs



class StyleLoss_blending(nn.Module):

    def __init__(self, original_imgs, weight):
        super(StyleLoss_blending, self).__init__()
        self.weight = weight
        self.Gs_orig = [gram_matrix(inp.unsqueeze(0)).detach() for inp in original_imgs]

    def forward(self, inputs):
        Gs = [gram_matrix(inp.unsqueeze(0)) for inp in inputs]
        self.loss = 0
        for i in range(len(Gs)):
            for j in range(len(self.Gs_orig)):
                # if i ==j:
                #     continue
                self.loss = torch.add(self.loss, F.mse_loss(Gs[i], Gs[j]))
        self.loss = self.weight*self.loss.div(len(Gs)**2)
        return inputs
