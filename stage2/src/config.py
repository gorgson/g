import argparse


def parse_args_llama():
    parser = argparse.ArgumentParser(description="llm")

    parser.add_argument("--model_name", type=str, default='llm')
    parser.add_argument("--project", type=str, default="instruction_tuning")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='Movies')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=5)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--resume", type=str, default='')

    # Model Training
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)
    parser.add_argument('--full_neighbor', type=bool, default=False, help='epochs')
    parser.add_argument('--nl', type=int, default=2, help='epochs')
    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=3)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--id", type=int, default=0)
    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='Llama-3.1-8B')
    parser.add_argument("--llm_model_path", type=str, default='/scratch/ys6310/llava-1.5-13b-hf')
    parser.add_argument("--llm_frozen", type=str, default='False')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--max_txt_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--pre_epoch", type=int, default=6)
    parser.add_argument("--k", type=int, default=10)
    # llm adapter
    parser.add_argument("--adapter_len", type=int, default=10)
    parser.add_argument("--adapter_layer", type=int, default=30)

    # distributed training parameters
    parser.add_argument("--log_dir", type=str, default='logs/')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--world_size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--gpu", default='0,1,2,3', type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument('--num_of_neighbors', type=int, default=7, help='e1pochs')
    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gat')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_out_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=512, help='Description of the parameter')
    parser.add_argument('--num_layers', type=int , default=1, help='epochs')
    args = parser.parse_args()
    return args