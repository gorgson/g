import os
import random
import torch
import dgl
import numpy as np
import pandas as pd
def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)  # If using DGL's randomness
    dgl.random.seed(seed)  # For additional DGL-specific randomness

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class LinkPredictionDataset(object):
    """
    一个包含 (train_graph, full_graph) 的数据集类：
      - train_graph: 只包含训练集正边
      - full_graph:  包含所有正边 (train + val + test)，有时调试/统计会用到

    同时生成 train/val/test 的正负样本边，以及对应标签。
    """
    def __init__(self,
                 root: str,
                 feat_name: str,
                 neg_sampling_ratio: float = 1.0,
                 verbose: bool = True,
                 device: str = "cpu",
                 do_subset_sampling: bool = True,
                 new_train_size: int = 3000,
                 new_val_size: int = 2000,
                 new_test_size: int = 1000):
        """
        root: 数据集所在文件夹
        feat_name: 特征文件名的前缀 (例如 'x' 就对应 x_feat.pt)
        neg_sampling_ratio: 负采样的比例 (默认 1:1)
        verbose: 是否打印信息
        device: 设备 ('cpu' 或 'cuda')
        do_subset_sampling: 是否要在拆分完成后从 train/val/test 再次抽样子集
        new_train_size: 需要的子训练集边数
        new_val_size:   需要的子验证集边数
        new_test_size:  需要的子测试集边数
        """
        root = os.path.normpath(root)
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        self.feat_name = feat_name
        self.device = torch.device("cuda")
        self.neg_sampling_ratio = neg_sampling_ratio

        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f"Feature name: {self.feat_name}")
            print(f"Device: {self.device}")
            print(f"Negative sampling ratio: {self.neg_sampling_ratio}")

        # 1) 加载边 (注意：所有边，不区分 train/val/test)
        edge_path = os.path.join(root, 'nc_edges-nodeid.pt')
        self.edge = torch.tensor(torch.load(edge_path), dtype=torch.int64).to(self.device)

        # 2) 加载节点特征
        feat_path = os.path.join(f'{self.name}_feature.pt')
        feat = torch.load(feat_path, map_location=self.device)
        self.num_nodes = feat.shape[0]

        # 3) full_graph：包含所有正边
        src, dst = self.edge.t()[0], self.edge.t()[1]
        self.full_graph = dgl.graph((src, dst), num_nodes=self.num_nodes).to(self.device)
        self.full_graph.ndata['feat'] = feat
        self.full_graph.ndata["index"]=torch.arange(self.num_nodes).cuda()
        if self.name=="Toys" or self.name=="Movies":
            df = pd.read_csv(f"{root}/{self.name}.csv")
            self.text=list(df["text"])
        else:
            graph_path = f"{root}/{self.name}_graph.dgl"
            graph = torch.load(graph_path, weights_only=True)
            self.text=[]
            for node_idx, node in graph["detail"].items():
                asin, desc, title, class_idx,reviews = node.values()
                self.text.append("title:"+title+"description"+"".join(desc)+"".join(reviews)[:512])
        # 4) 拆分边，并进行负采样（确保训练/验证/测试负边不重叠）
        self._prepare_edges()

        # 5) 基于训练集「正边」构造 train_graph
        #    注意这里不包含验证集 / 测试集正边，以保证只训练可见的正边
        train_pos_edges = self.train_edges[self.train_labels == 1]
        train_src, train_dst = train_pos_edges.t()
        self.train_idx= torch.unique(torch.cat([train_src, train_dst]))
        self.train_graph = dgl.graph((train_src, train_dst), num_nodes=self.num_nodes).to(self.device)
        self.train_graph.ndata['feat'] = feat
        self.train_graph.ndata["index"]=torch.arange(self.num_nodes).cuda()

        # 6) （可选）从 train/val/test 采样更小规模的边做最终训练/验证/测试
        #    此时 train_graph 仍是使用全部训练正边构造，以便 GNN 充分进行信息传递
        if do_subset_sampling:
            self._sample_edge_subsets(
                new_train_size=new_train_size,
                new_val_size=new_val_size,
                new_test_size=new_test_size
            )
        if self.name=="Movies":
            target_domain='Movies & TV'
            co="co-viewed"
        elif self.name=="Toys":
            target_domain='Toys & Games'
            co="co-buyed"
        elif self.name=="9_CDs_and_Vinyl":
            target_domain='Music CDs'
            co="co-buyed"
        elif self.name=="4_Arts_Crafts_and_Sewing":
            target_domain='Arts_Crafts_and_Sewing' 
            co="co-buyed"
        self.prompt=[
            f"I’m starting a link prediction task in the Amazon-{target_domain} dataset. Each node represents a {target_domain} with text and image features, and edges indicate {co} relation. Given the two nodes: Node 1: The text features are ",
            #<text sequence>
            ", and the image features are ",
            #<img sequence>.
            "Node 2: The text features are ",
            #<text sequence>
            ", and the image features are ",
            #<img sequence>
            ".The neighbors of node 1 are described in the following template: <text feature>, <image feature>. It has the following neighbors at hop 1 (Directly connected): ",
            #<1-hop neighbor 1 text feature>, <1-hop neighbor 1 image feature>
            "It has the following neighbors at hop 2 (Indirectly connected by shared neighbors): ",
            #<2-hop neighbor 1 text feature>, <2-hop neighbor 1 image feature>
            '''Based on the information provided, please determine whether a link exists between the two nodes. Answer "yes" if a link exists or "no" if it does not.'''
        ]

    def _prepare_edges(self):
        """把所有正边拆分到 train/val/test，并对每个集合做负采样，再合并出 (edges, labels)。"""
        # 1) 随机打乱所有正边
        edges = self.edge.tolist()
        random.shuffle(edges)
        num_edges = len(edges)

        # 2) 按比例拆分为 train/val/test (6:2:2)
        train_size = int(num_edges * 0.6)  # 60%
        val_size = int(num_edges * 0.2)    # 20%
        test_size = num_edges - train_size - val_size  # 20%

        train_edges = edges[:train_size]
        val_edges = edges[train_size:train_size + val_size]
        test_edges = edges[train_size + val_size:]

        # 3) 计算各自所需负边数量
        train_neg_needed = int(len(train_edges) * self.neg_sampling_ratio)
        val_neg_needed = int(len(val_edges) * self.neg_sampling_ratio)
        test_neg_needed = int(len(test_edges) * self.neg_sampling_ratio)
        total_neg_needed = train_neg_needed + val_neg_needed + test_neg_needed

        # 4) 一次性采样所有负边，保证不重叠
        all_neg_edges = self._sample_negative_edges(total_neg_needed)

        # 5) 拆分负边到 train/val/test
        train_neg = all_neg_edges[:train_neg_needed]
        val_neg = all_neg_edges[train_neg_needed:train_neg_needed + val_neg_needed]
        test_neg = all_neg_edges[train_neg_needed + val_neg_needed:]

        # 6) 合并正负边 + 构造标签 (1表示正样本，0表示负样本)
        self.train_edges = torch.cat([
            torch.tensor(train_edges, dtype=torch.int64, device=self.device),
            train_neg
        ], dim=0)
        self.train_labels = torch.cat([
            torch.ones(len(train_edges), device=self.device),
            torch.zeros(train_neg_needed, device=self.device)
        ], dim=0)

        self.val_edges = torch.cat([
            torch.tensor(val_edges, dtype=torch.int64, device=self.device),
            val_neg
        ], dim=0)
        self.val_labels = torch.cat([
            torch.ones(len(val_edges), device=self.device),
            torch.zeros(val_neg_needed, device=self.device)
        ], dim=0)

        self.test_edges = torch.cat([
            torch.tensor(test_edges, dtype=torch.int64, device=self.device),
            test_neg
        ], dim=0)
        self.test_labels = torch.cat([
            torch.ones(len(test_edges), device=self.device),
            torch.zeros(test_neg_needed, device=self.device)
        ], dim=0)

    def _sample_negative_edges(self, total_neg_needed: int):
        """
        一次性从图中不存在的边中采样 total_neg_needed 条负边，保证无重复。
        """
        neg_edges = set()
        # 包含所有正边 (u, v)，以防采到正边
        all_edges = set(map(tuple, self.edge.tolist()))

        while len(neg_edges) < total_neg_needed:
            u = random.randint(0, self.num_nodes - 1)
            v = random.randint(0, self.num_nodes - 1)
            # 保证 (u,v) 不在正边里，且 u != v，且不重复
            if u != v and (u, v) not in all_edges and (u, v) not in neg_edges:
                neg_edges.add((u, v))

        # 变成列表后再随机打乱一下，以免后面切分出现顺序相关问题
        neg_edges_list = list(neg_edges)
        random.shuffle(neg_edges_list)

        return torch.tensor(neg_edges_list, dtype=torch.int64, device=self.device)

    def _sample_edge_subsets(self, new_train_size=3000, new_val_size=2000, new_test_size=1000):
        """
        从原本的 train_edges/val_edges/test_edges 中随机采样指定数量的边，
        并相应缩减 train_labels/val_labels/test_labels。
        这样可以得到更小的(训练/验证/测试)集做实验。
        
        注：train_graph 保持不变，仍包含所有训练正边，用于 GNN 消息传递。
        """

        # 采样训练集
        original_train_size = len(self.train_edges)
        if new_train_size > original_train_size:
            print(f"[Warning] 需要的训练集大小({new_train_size})超过了原始训练集({original_train_size})，将使用全部训练集。")
            new_train_size = original_train_size
        train_idx = random.sample(range(original_train_size), new_train_size)
        train_idx = torch.tensor(train_idx, dtype=torch.long, device=self.device)
        self.train_edges = self.train_edges[train_idx]
        self.train_labels = self.train_labels[train_idx]

        # 采样验证集
        original_val_size = len(self.val_edges)
        if new_val_size > original_val_size:
            print(f"[Warning] 需要的验证集大小({new_val_size})超过了原始验证集({original_val_size})，将使用全部验证集。")
            new_val_size = original_val_size
        val_idx = random.sample(range(original_val_size), new_val_size)
        val_idx = torch.tensor(val_idx, dtype=torch.long, device=self.device)
        self.val_edges = self.val_edges[val_idx]
        self.val_labels = self.val_labels[val_idx]

        # 采样测试集
        original_test_size = len(self.test_edges)
        if new_test_size > original_test_size:
            print(f"[Warning] 需要的测试集大小({new_test_size})超过了原始测试集({original_test_size})，将使用全部测试集。")
            new_test_size = original_test_size
        test_idx = random.sample(range(original_test_size), new_test_size)
        test_idx = torch.tensor(test_idx, dtype=torch.long, device=self.device)
        self.test_edges = self.test_edges[test_idx]
        self.test_labels = self.test_labels[test_idx]

        if self.verbose:
            print(f"已从原始训练集抽取 {new_train_size} 条边，验证集 {new_val_size} 条边，测试集 {new_test_size} 条边。")

    def get_edge_split(self):
        """返回 {train:..., val:..., test:...}，其中包含 (edges, labels)。"""
        return {
            'train': {
                'edges': self.train_edges,
                'labels': self.train_labels
            },
            'val': {
                'edges': self.val_edges,
                'labels': self.val_labels
            },
            'test': {
                'edges': self.test_edges,
                'labels': self.test_labels
            }
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}(name={self.name}, "
                f"nodes={self.num_nodes}, edges={self.edge.shape[0]})")
