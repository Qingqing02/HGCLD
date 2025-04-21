import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret


def reg_pick_embeds(embeds_list):
	reg_loss = 0
	for embeds in embeds_list:
		reg_loss += embeds.square().sum()
	return reg_loss


def contrast(nodes, allEmbeds, allEmbeds2=None):
	if allEmbeds2 is not None:
		pckEmbeds = allEmbeds[nodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
	else:
		uniqNodes = t.unique(nodes)
		pckEmbeds = allEmbeds[uniqNodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
	return scores



def cal_infonce_loss(embeds1, embeds2, temp):

        embeds1 = F.normalize(embeds1 + 1e-8, p=2)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2)
        pckEmbeds1 = embeds1
        pckEmbeds2 = embeds2
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
        return -t.log(nume / deno)



def cal_infonce_loss_spec_nodes(embeds1, embeds2, nodes, temp):

        embeds1 = F.normalize(embeds1 + 1e-8, p=2)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
        return -t.log(nume / deno).mean()