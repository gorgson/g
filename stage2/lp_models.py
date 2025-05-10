import torch
import torch.nn.functional as F
import dgl.nn as dglnn
import tqdm
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.nn import GATv2Conv, GATConv
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Linear

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout=0.5, use_sigmoid=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_sigmoid = use_sigmoid

        # 创建 GCN 层
        self.layers = nn.ModuleList()
        if num_layers < 1:
            raise ValueError("num_layers 必须 >= 1")

        # 如果只有 1 层，直接从 in_size 到 out_size
        if num_layers == 1:
            self.layers.append(
                dglnn.GraphConv(in_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
            )
        else:
            # 第一层: in_size -> hid_size
            self.layers.append(
                dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
            )
            # 中间 (num_layers - 2) 层: hid_size -> hid_size
            for _ in range(num_layers - 2):
                self.layers.append(
                    dglnn.GraphConv(hid_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
                )
            # 最后一层: hid_size -> out_size
            self.layers.append(
                dglnn.GraphConv(hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, src, dst):
        # 取得节点特征
        h = graph.ndata['feat']  # [num_nodes, in_size]

        # 依次经过每一层 GCN
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)  # [num_nodes, out_dim_of_this_layer]
            # 在除最后一层外都加上激活和 dropout
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)

        # 取出源节点和目的节点的 embedding
        src_h = h[src]  # [batch_size, out_size]
        dst_h = h[dst]  # [batch_size, out_size]

        # 点积打分
        edge_scores = torch.sum(src_h * dst_h, dim=-1)  # [batch_size]

        # 可选择是否对打分做 Sigmoid
        if self.use_sigmoid:
            edge_scores = torch.sigmoid(edge_scores)

        return edge_scores

class MLP(nn.Module):
    """
    纯 MLP 用于链接预测，直接拼接 (src, dst) 的原始特征 [in_size*2 -> ... -> 1]。
    """
    def __init__(self, in_size, hid_size, out_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # 注意：这里 in_size 指的是节点特征维度
        # 由于 MLP 不做图卷积，需要 (src_feat + dst_feat) => in_size*2
        if num_layers == 1:
            self.layers.append(nn.Linear(in_size * 2, out_size))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_size * 2, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        elif num_layers == 3:
            self.layers.append(nn.Linear(in_size * 2, hid_size))
            self.layers.append(nn.Linear(hid_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        else:
            raise ValueError(f"Unsupported num_layers={num_layers}, must be 1/2/3.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, src, dst):
        src_h = graph.ndata['feat'][src]
        dst_h = graph.ndata['feat'][dst]
        edge_features = torch.cat([src_h, dst_h], dim=-1)
        h = edge_features
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return torch.sigmoid(h).squeeze()

class SAGE(nn.Module):

    def __init__(self,
                 in_size,
                 hid_size,
                 out_size,
                 num_layers=2,
                 aggregator_type='mean',
                 dropout=0.5,
                 use_sigmoid=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_sigmoid = use_sigmoid
        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("num_layers 必须 >= 1")

        # 如果只有 1 层，直接 in_size -> out_size
        if num_layers == 1:
            self.layers.append(
                dglnn.SAGEConv(in_size, out_size, aggregator_type=aggregator_type, feat_drop=0.5)
            )
        else:
            # 第 1 层: in_size -> hid_size
            self.layers.append(
                dglnn.SAGEConv(in_size, hid_size, aggregator_type=aggregator_type, feat_drop=0.5)
            )
            # 中间层: hid_size -> hid_size (若 num_layers > 2)
            for _ in range(num_layers - 2):
                self.layers.append(
                    dglnn.SAGEConv(hid_size, hid_size, aggregator_type=aggregator_type, feat_drop=0.5)
                )
            # 最后一层: hid_size -> out_size
            self.layers.append(
                dglnn.SAGEConv(hid_size, out_size, aggregator_type=aggregator_type, feat_drop=0.5)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, src, dst):
        h = graph.ndata['feat']

        # 依次经过每一层
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
        src_h = h[src]  # [batch_size, out_size]
        dst_h = h[dst]  # [batch_size, out_size]

        # 点积打分
        edge_scores = torch.sum(src_h * dst_h, dim=-1)  # [batch_size]

        # 是否加 Sigmoid
        if self.use_sigmoid:
            edge_scores = torch.sigmoid(edge_scores)

        return edge_scores

class GATv2(nn.Module):

    def __init__(self,
                 in_size,
                 hid_size,
                 out_size,
                 num_layers,
                 heads,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=False,
                 dropout=0.5,
                 use_sigmoid=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_sigmoid = use_sigmoid

        # 若 heads 是单个 int，则复制成 list
        if isinstance(heads, int):
            heads = [heads] * num_layers
        assert len(heads) == num_layers, "heads 的长度需与 num_layers 相同"
        self.heads = heads

        self.gatv2_layers = nn.ModuleList()
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        if num_layers < 1:
            raise ValueError("num_layers 必须 >= 1")

        # 若只有 1 层，直接 in -> out
        if num_layers == 1:
            self.gatv2_layers.append(
                GATv2Conv(in_size, out_size, num_heads=heads[0],
                          feat_drop=feat_drop,
                          attn_drop=attn_drop,
                          negative_slope=negative_slope,
                          residual=residual,
                          activation=None,  # 最后一层不加激活，在外部自行处理
                          allow_zero_in_degree=True)
            )
        else:
            # 第1层: in_size -> hid_size
            self.gatv2_layers.append(
                GATv2Conv(in_size, hid_size, num_heads=heads[0],
                          feat_drop=feat_drop,
                          attn_drop=attn_drop,
                          negative_slope=negative_slope,
                          residual=False,  # 第一层可不加残差
                          activation=self.activation,
                          allow_zero_in_degree=True)
            )
            # 中间层: hid_size -> hid_size
            for l in range(num_layers - 2):
                self.gatv2_layers.append(
                    GATv2Conv(hid_size * heads[l],  # 上一层输出 hid_size * heads[l]
                              hid_size,
                              num_heads=heads[l+1],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              negative_slope=negative_slope,
                              residual=residual,
                              activation=self.activation,
                              allow_zero_in_degree=True)
                )
            # 最后一层: hid_size -> out_size
            self.gatv2_layers.append(
                GATv2Conv(hid_size * heads[num_layers - 2],
                          out_size,
                          num_heads=heads[num_layers - 1],
                          feat_drop=feat_drop,
                          attn_drop=attn_drop,
                          negative_slope=negative_slope,
                          residual=residual,
                          activation=None,  # 最后一层一般不再加激活
                          allow_zero_in_degree=True)
            )

    def forward(self, graph, src, dst):
        """
        graph: DGLGraph，其中 graph.ndata['feat'] 是节点特征
        src, dst: 需要预测连边的 (源, 目的) 节点 ID
        """
        # 节点特征
        h = graph.ndata['feat']

        for i, layer in enumerate(self.gatv2_layers):
            # GATv2Conv 输出形状一般是 [N, out_dim, num_heads] (如果 activation=None)
            # 或直接是 [N, out_dim * num_heads] (取决于实现)
            # 这里我们先得到 (N, out_dim, num_heads)，再 flatten(1) 拼成 [N, out_dim * num_heads]
            # 以便后续层继续处理。
            h = layer(graph, h)  # 形状: [N, out_dim, num_heads] 或 [N, out_dim * num_heads]

            # 若激活函数是 None，则可能是 [N, out_dim, heads]，需要手动 flatten
            # 这里的做法：先把多头拼起来 => flatten(1)
            # 注意：如果 GATv2Conv 实现已经自动 flatten，那么可以跳过
            if i != self.num_layers - 1:
                # 中间层：激活 + dropout
                # 若 layer 带有激活，会返回 [N, out_dim, heads]
                if h.dim() == 3:
                    N, out_dim, num_heads = h.shape
                    h = h.view(N, out_dim * num_heads)  # flatten
                h = self.dropout(h)
                h = self.activation(h)
            else:
                # 最后一层（通常无激活），只做 flatten
                if h.dim() == 3:
                    N, out_dim, num_heads = h.shape
                    h = h.view(N, out_dim * num_heads)

        # 取出连边两端节点的 embedding
        src_h = h[src]
        dst_h = h[dst]

        # 点积打分
        edge_scores = torch.sum(src_h * dst_h, dim=-1)
        if self.use_sigmoid:
            edge_scores = torch.sigmoid(edge_scores)

        return edge_scores
