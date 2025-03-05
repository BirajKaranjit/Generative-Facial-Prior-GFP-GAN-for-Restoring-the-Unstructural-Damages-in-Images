#utils.py
import torch
from data_loader import *
from functools import partial
from torch.utils.data.dataloader import default_collate

#L2 Normalization helps stabilize training by ensuring all feature vectors have a uniform scale.
def l2_norm(input,axis=1):  #normalized the input tensor
    norm = torch.norm(input,2,axis,True) #Computes the L2 norm (Euclidean norm) of input along the specified axis.
    output = torch.div(input, norm) #Element-wise division of input by norm to normalize it.
    return output


# simplified my_collate function  #If our dataset does not contain any 'None' values then we can use it
def my_collate(batch):
    return torch.utils.data.dataloader.default_collate(batch)

def requires_grad(model, flag=True):
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad = flag

def need_grad(x):
    x = x.detach()
    x.requires_grad_()
    return x

def init_weights(model,init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    model.apply(init_func)

#This accumulate() function performs exponential moving average (EMA) updates between two models, model1 and model2
def accumulate(model1, model2, decay=0.999,use_buffer=False):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    if use_buffer:
        for p1,p2 in zip(model1.buffers(),model2.buffers()):
            p1.detach().copy_(decay*p1.detach()+(1-decay)*p2.detach())

def setup_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_state_dict_with_prefix_removal(model, state_dict_path, device="cpu"):
    """Loads a state dictionary, removing 'module.' prefixes if present.

    Args:
        model (torch.nn.Module): The model to load the state dictionary into.
        state_dict_path (str): The path to the state dictionary.
        device (str): Device to load the state_dict to. Default: "cpu"
    """
    state_dict = torch.load(state_dict_path, map_location=device)

    # Check if any keys have the 'module.' prefix
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
             new_key = key.replace('module.', '')
             new_state_dict[new_key] = value
        state_dict = new_state_dict
        print("Removed 'module.' prefix from state_dict keys.")

    model.load_state_dict(state_dict)


def get_data_loader(args):

    train_data = GFPData(dist=args.dist,
                    mean=args.mean,
                    std=args.std,
                    train_hq_root=args.train_hq_root,
                    train_lq_root=args.train_lq_root,
                    size=args.size,
                    # eye_enlarge_ratio=args.eye_enlarge_ratio,
                    eval=False)

    test_data = None
    if args.eval:
        test_data = GFPData(dist=args.dist,
                    mean=args.mean,
                    std=args.std,
                    val_hq_root=args.val_hq_root,    #only for evaluation
                    val_lq_root=args.val_lq_root,
                    size=args.size,
                    eval=True )

    use_collate = partial(my_collate,dataset=train_data)  #functools.partial is a Python function that allows you to fix certain arguments of a function while keeping others flexible. This means that when calling use_collate, you only need to provide the remaining arguments.

    train_loader = torch.utils.data.DataLoader(train_data,
                        collate_fn= my_collate,  # i used simple default_collate()
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True) #####myadd just test


    test_loader = None if test_data is None else \
        torch.utils.data.DataLoader(
                        test_data,
                        collate_fn = my_collate,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    return train_loader,test_loader,len(train_loader)


#he function merge_args(args, params) is used to update an existing argument object (args) with values from another object (params).
def merge_args(args,params):
   for k,v in vars(params).items():  #vars(params).items() extracts all attributes and their values from params as key-value pairs.
      setattr(args,k,v) #The setattr(args, k, v) function dynamically sets the attribute k in args to the value v.
   return args

def convert_img(img,unit=False):

    img = (img + 1) * 0.5
    if unit: #If unit=True, the function scales pixel values to [0,255], which is the standard range for 8-bit images (used for saving or displaying).
      return torch.clamp(img*255+0.5,0,255)

    return torch.clamp(img,0,1) ##If unit=False, the function keeps values in [0,1], which is useful for processing images in PyTorch.


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b,c,h,w = flow.shape
    flow_norm = 2 * torch.cat([flow[:,:1,...]/(w-1),flow[:,1:,...]/(h-1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0,2,3,1)
    return deformation

def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """
    b,c,h,w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed


def warp_image(source_image, deformation):
    r"""warp the input image according to the deformation

    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation)


def add_list(x):
    r = None

    for v in x:
        r = v if r is None else r + v
    return r

def compute_cosine(x):
    mm = np.matmul(x,x.T)
    norm = np.linalg.norm(x,axis=-1,keepdims=True)
    dis = mm / np.matmul(norm,norm.T)
    return  dis - np.eye(*dis.shape)

def compute_graph(cos_dis):
    index = np.where(np.triu(cos_dis) >= 0.68)

    dd = {}
    vis = {}


    for i in np.unique(index[0]):

        if i not in vis:
            vis[i] = i
            dd[vis[i]] = [i]

        for j in index[1][index[0]==i]:
            if j not in vis:
                vis[j] = vis[vis[i]]
                dd[vis[i]] = dd.get(vis[i],[]) + [j]


            elif j in vis and vis[vis[j]] != vis[vis[i]]:
                old_root = vis[vis[j]]
                for v in dd[vis[vis[j]]]:
                    dd[vis[vis[i]]] += [v]
                    vis[v] = vis[vis[i]]
                del dd[old_root]

    for k,v in dd.items():
        dd[k] = list(set(v+[k]))
    return dd,index