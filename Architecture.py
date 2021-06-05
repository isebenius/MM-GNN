from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, GATConv, TAGConv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GraphConv
from torch_geometric.nn import BatchNorm
from MMPool import SAGPool_ROI,MMPool
from MMTopKPool import TopKPooling, MMTopKPool

class MMGNN(torch.nn.Module):
    def __init__(self, hidden_channels,ratio = 0.8):
        super(MMGNN, self).__init__()
        #torch.manual_seed(12345)

        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(dataset2.num_node_features, hidden_channels)

        #choose desired multimodal pooling layer here
        self.pool_double1 = MMTopKPool(hidden_channels, ratio = ratio)
        #self.pool_double1 = MMPool(hidden_channels, ratio = ratio, num_ROIs = 273)

        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)

        self.norm1 = BatchNorm(hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)

        self.norm3 = BatchNorm(hidden_channels)
        self.norm4 = BatchNorm(hidden_channels)

        self.norm5 = BatchNorm(int(4*hidden_channels))

        self.lin_single = Linear(int(2*hidden_channels), int(dataset.num_classes))
        self.lin_double = Linear(int(4*hidden_channels), int(dataset.num_classes))

    def forward(self, x1, edge_index1, x2= None, edge_index2 = None, edge_attr1 = None, edge_attr2 = None, batch=None):
        #print(x, batch)

        x1 = self.conv1(x1, edge_index1)#, edge_weight = edge_attr1)
        x2 = self.conv2(x2, edge_index2)#, edge_weight = edge_attr2)

        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        x1 = x1.relu()
        x2 = x2.relu()

        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, batch, perm, score, _, _ = self.pool_double1(x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, batch = batch)

        x1 = self.conv3(x1, edge_index1)#, edge_weight = edge_attr1)
        x2 = self.conv4(x2, edge_index2)#, edge_weight = edge_attr2)
        #print(x1[0])

        x1 = self.norm3(x1)
        x2 = self.norm4(x2)
        
        x1 = x1.relu()
        x2 = x2.relu()

        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.dropout(x2, p=0.2, training=self.training)


        x1 = torch.cat([global_max_pool(x1, batch), global_mean_pool(x1, batch)], dim=1)
        x2 = torch.cat([global_max_pool(x2, batch), global_mean_pool(x2, batch)], dim=1)

        x = torch.cat((torch.atleast_2d(x1),torch.atleast_2d(x2)),dim = 1)

        x = self.norm5(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin_double(x)
        x = torch.softmax(x, 1).squeeze(1)

        
        return x, score