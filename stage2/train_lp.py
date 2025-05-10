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
from src.model import load_model_lp, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
import sys 
from lp_dataset import LinkPredictionDataset
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
    dataset = LinkPredictionDataset(
          root=os.path.join(data_path, dataset_name),
          feat_name=None,
          neg_sampling_ratio=1.0,
          device=device
      )
    prompt=dataset.prompt
    train_data = dataset.get_edge_split()['train']
    val_data = dataset.get_edge_split()['val']
    test_data= dataset.get_edge_split()['test']
    train_edges, train_labels = train_data['edges'], train_data['labels']
    val_edges, val_labels = val_data['edges'], val_data['labels']
    test_edges, test_labels = test_data['edges'], test_data['labels']
    train_loader = torch.utils.data.DataLoader(
        dataset=list(zip(train_edges, train_labels)),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=list(zip(val_edges, val_labels)),
        batch_size=args.eval_batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=list(zip(test_edges, test_labels)),
        batch_size=args.test_batch_size,
        shuffle=True
    )
    


    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model_lp[args.model_name](prompt=prompt,text=dataset.text,args=args,g=None,train_idx=dataset.train_graph,dataset=dataset)
    #model.load_state_dict(torch.load("lp3trans.pth"))
    '''
    checkpoint_path =f"lp{str(args.id)}trans.pth"
    if args.id==1 :
        if args.dataset=="Movies":
            pass
        else:
            model.load_state_dict(torch.load(checkpoint_path))
            print("load")
            print(checkpoint_path) 
    else:
        if args.dataset=="Movies":
            model.load_state_dict(torch.load(f"lp{str(args.id-1)}trans.pth"))
            print("load")
            print(f"{str(args.id-1)}trans.pth") 
        else:
            model.load_state_dict(torch.load(checkpoint_path))
            print("load")
            print(checkpoint_path) 
    '''
    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd},],
        betas=(0.9, 0.95)
    )

    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    best_val_loss, best_val_acc = float('inf'), -float('inf')
    best_epoch = 0
    
    model.feature=torch.load(f'{args.dataset}_feature.pt')
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
        if epoch%1==0:
            '''
            if args.dataset=="4_Arts_Crafts_and_Sewing":
              torch.save(model.state_dict(), f"lp{str(args.id+1)}trans.pth")
              print("save") 
              print(f"{str(args.id+1)}basetrans.pth")
            else:
              torch.save(model.state_dict(), checkpoint_path)
              print("save") 
              print(checkpoint_path)
            '''
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
    
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating
    model = _reload_best_model(model, args)

    model.eval()
    eval_output = []
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)


    # Step 6. Post-processing & c
    path = f'{args.output_dir}/{args.dataset}_{args.model_name}_{args.llm_model_name}_seed{seed}.csv'
    acc = eval_funcs[args.dataset](eval_output, path)
    print(f'Test Acc {acc}')
    wandb.log({'Test Acc': acc})
    
if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()