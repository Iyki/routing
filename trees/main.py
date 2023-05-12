

# %%capture captured
import generator
import model
import pickle
import torch
from fastbook import *
from fastai.collab import *
from fastai.tabular.all import *
from fastai.losses import *
import torch.nn as nn


# todo: change sz to number of bits
def get_small_emb_sz(dls_df, n_bits=8):
    emb = get_emb_sz(dls_df)
    
    return [(emb[0][0], n_bits)]


def MRELoss(inp, targ) -> Tensor:
    inp = torch.flatten(inp).float()
    targ = torch.flatten(targ).float()
    nom = torch.nn.functional.l1_loss(inp, targ)
    noo = torch.Tensor([0]).repeat(targ.size(0)).to(device)
    denom = torch.nn.functional.l1_loss(noo, targ)
    loss = (nom / denom).mean()
    return loss


def CombineLoss(inp, targ) -> Tensor:
    mse_loss = MSELossFlat()
    loss_1 = mse_loss(inp, targ)

    loss_2 = MRELoss(inp, targ)

    return alpha * loss_1 + (1 - alpha) * loss_2


lg_N = 4
num_epochs = 100
alpha = 0.5
loss_function = CombineLoss
# emb_sz = 1
n_bits = 8

'''
Data can either be generated from scratch or read from a file that it was previously saved to
'''

savepath = 'save/' + str(alpha) + '_' + str(lg_N) + '_' + str(num_epochs) + '/'
if (not os.path.exists(savepath)):
    os.makedirs(savepath)

# dataset = generate_dataset(2**lg_N)

real_graph_paths = [
    "/jumbo/lisp/ike/code/DistanceLabelling/datasets/ENZYMES_g1/ENZYMES_g1.edges"]
full_dataset = generator.generate_real_graphs_dataset(real_graph_paths)
dataset = generator.get_edge_list(real_graph_paths[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.array(dataset)
print(data.shape)

full_data = np.array(full_dataset)
# if loading from csv, cut the first column out (line numbers)
# df = pd.DataFrame(data[:, 1:], columns=['src', 'dst', 'label'])
# node_pairs = pd.read_pickle(savepath + 'node_pairs.pkl')
# node_id = pd.read_pickle(savepath + 'node_id.pkl')

# otherwise (if generated), use data directly
df = pd.DataFrame(data, columns=['src', 'dst', 'label'])

dls_df = CollabDataLoaders.from_df(df, bs=64)
embs = get_small_emb_sz(dls_df, n_bits)
max_value = df.label.max() * embs[0][0] # df.label.max()
trainer = model.CollabNN(*embs, y_range=(0, max_value))
# learn = Learner(dls_df, trainer, loss_func = loss_function, path=savepath)

learn = Learner(dls_df, trainer, loss_func=loss_function,
                path=savepath, metrics=[mse, mae, MRELoss])

# torch.save(trainer, modelpath + 'model.pth')


# learn.load('model')
SaveModelCallback()  # remove? with_opt=True),
learn.remove_cb(ProgressCallback)
learn.fit_one_cycle(n_epoch=num_epochs, lr_max=5e-3, wd=0.01) 
# , cbs=[ShowGraphCallback(), EarlyStoppingCallback(min_delta=0.01, patience=250)])
# torch.save(trainer, savepath + 'model.pth')


allnodes = set(df.src).union(set(df.dst))
# print(allnodes)
embs = learn.model.save_embeddings(Tensor(list(allnodes)).to(device).int())
print(embs)
with open(savepath + "embeddings.pkl", "wb") as outfile:
    pickle.dump(embs, outfile)

print(full_data[:10,:2].shape)
print(learn.model(Tensor([[2, 30]]).to(device).int()))
