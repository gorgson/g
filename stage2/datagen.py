import torch
import wandb
import copy
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import dgl
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
import copy
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from torch.nn.utils import clip_grad_norm_
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
import sys 
sys.path.append("..") 
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
def load_bert(arg):
    config = TuringNLRv3Config.from_pretrained(
        "config.json",
        output_hidden_states=True)
    from src.models.modeling_graphformers import GraphFormersForNeighborPredict
    config.hidden_size=arg.hidden_dim
    model = GraphFormersForNeighborPredict(config,arg)
        # model.load_state_dict(torch.load(args.model_name_or_path, map_location="cpu")['model_state_dict'], strict=False)

    return model
def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping

def main(args):

    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args,
               mode="offline")
    #pre_model=load_bert(args)
    #pre_model.load_state_dict(torch.load(f'../GRA/nc/check/check_{args.dataset}{args.pre_epoch}_best.pth'))
    #for name, param in pre_model.named_parameters():
        #param.requires_grad = False
    seed_everything(seed=args.seed)
    data_path = '/scratch/ys6310/Mario/dataset/' # replace this with the path where you save the datasets
    dataset_name = args.dataset
    verbose = True
    device = torch.device('cpu')
    if dataset_name =="Movies" or  dataset_name =="Toys":
        from nnc_dataset import NodeClassificationDataset, NodeClassificationEvaluator
        dataset = NodeClassificationDataset(
            root=os.path.join(data_path, dataset_name),
            data_path=data_path,
            verbose=verbose,
            device=device,
            save=False
        ) 
    else:
        from nnnc_dataset import NodeClassificationDataset, NodeClassificationEvaluator
        dataset = NodeClassificationDataset(
            root=os.path.join(data_path, dataset_name),
            data_path=data_path,
            verbose=verbose,
            device=device,
            save=False
        ) 
    prompt=dataset.prompt
    map=dataset.label_to_second_category
    g = dataset.graph
    g = dgl.remove_self_loop(g).to("cpu")
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = dgl.add_self_loop(g)
    num_classes=dataset.num_classes

    splits ={}         
    train_idx = g.ndata['train_mask'].nonzero().long().squeeze(-1)
    val_idx = g.ndata['val_mask'].nonzero().long().squeeze(-1)
    test_idx = g.ndata['test_mask'].nonzero().long().squeeze(-1)
    if args.full_neighbor:

        sampler = MultiLayerFullNeighborSampler(num_layers=cfg.num_layers, prefetch_node_feats=['text_feat','image_feat'])
    else:
       
        sampler = NeighborSampler([args.num_of_neighbors] * args.num_layers, prefetch_node_feats=['text_feat','image_feat'])

    use_uva=True
    train_loader = DataLoader(
        g,
        train_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    val_loader = DataLoader(
        g,
        val_idx,
        sampler,
        batch_size=args.eval_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    test_loader = DataLoader(
        g,
        test_idx,
        sampler,
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )


    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](prompt=prompt,text=dataset.text,map=map,args=args,g=g,train_idx=train_idx)
    checkpoint_path =f"{str(args.id)}trans.pth"
    if args.id==1 :
        if args.dataset=="Movies":
            pass
        else:
            model.load_state_dict(torch.load(checkpoint_path))
            print("load")
            print(checkpoint_path) 
    else:
        if args.dataset=="Movies":
            model.load_state_dict(torch.load(f"{str(args.id-1)}trans.pth"))
            print("load")
            print(f"{str(args.id-1)}trans.pth") 
        else:
            model.load_state_dict(torch.load(checkpoint_path))
            print("load")
            print(checkpoint_path) 

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd},],
        betas=(0.9, 0.95)
    )

    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    # Step 5. Training
    num_training_steps = len(train_loader)
    best_val_loss, best_val_acc = float('inf'), -float('inf')
    best_epoch = 0

    model.feature=torch.load(f'{args.dataset}_data_feature.pt')
    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.
     
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
        
            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.


        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})
        if  epoch%1==0:
          if args.dataset=="9_CDs_and_Vinyl":
              torch.save(model.state_dict(), f"{str(args.id+1)}trans.pth")
              print("save") 
              print(f"{str(args.id+1)}trans.pth")
          else:
              torch.save(model.state_dict(), checkpoint_path)
              print("save") 
              print(checkpoint_path)
          val_loss = 0
          eval_output = []
          model.eval()
          with torch.no_grad():
              for step, batch in enumerate(val_loader):
                  loss = model.val_forward(batch)
                  val_loss += loss.item()
              val_loss = val_loss/len(val_loader)
              print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
              wandb.log({'Val Loss': val_loss})

          if val_loss < best_val_loss:
              best_val_loss = val_loss
              _save_checkpoint(model, optimizer, epoch, args, is_best=True)
              best_epoch = epoch

          print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

          if epoch - best_epoch >= args.patience:
              print(f'Early stop at epoch {epoch}')
              break






if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()