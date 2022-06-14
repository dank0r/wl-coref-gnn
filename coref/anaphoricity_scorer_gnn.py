""" Describes AnaphicityScorer, a torch module that for a matrix of
mentions produces their anaphoricity scores.
"""
import torch

from torch_geometric.nn import GCNConv, GATConv
from coref import utils
from coref.config import Config

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=60):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels, out_channels, edge_dim=edge_dim)
        # self.softmax = torch.nn.Softmax(dim=1)

    def encode(self, x, edge_index, edge_features):
        x = self.conv1(x, edge_index, edge_features).relu()
        return self.conv2(x, edge_index, edge_features)

    def decode(self, z, edge_label_index):
        # [2*batch_size*n_ants]
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index, edge_features, batch_size):
        z = self.encode(x, edge_index, edge_features)
        res = self.decode(x, edge_index)
        # sum up the score for adjacent edges
        res_half = res[:res.shape[0]//2]
        res_half += res[res.shape[0]//2:]
        logits = res_half.reshape((batch_size, -1))
        return logits


class AnaphoricityScorer(torch.nn.Module):
    """ Calculates anaphoricity scores by passing the inputs into a FFNN """
    def __init__(self,
                 in_features: int,
                 config: Config):
        super().__init__()
        hidden_size = config.hidden_size
        if not config.n_hidden_layers:
            hidden_size = in_features
        layers = []
        self.gnn = Net(config.hidden_size, 64, 64)
        for i in range(config.n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_size if i else in_features,
                                           hidden_size),
                           torch.nn.LeakyReLU(),
                           torch.nn.Dropout(config.dropout_rate)])
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_size, out_features=1)

    def forward(self, *,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                all_mentions: torch.Tensor,
                mentions_batch: torch.Tensor,
                pw_batch: torch.Tensor,
                top_indices_batch: torch.Tensor,
                top_rough_scores_batch: torch.Tensor,
                current_i: int,
                nominal_batch_size: int
                ) -> torch.Tensor:
        """ Builds a pairwise matrix, scores the pairs and returns the scores.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb]
            mentions_batch (torch.Tensor): [batch_size, mention_emb]
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb]
            top_indices_batch (torch.Tensor): [batch_size, n_ants]
            top_rough_scores_batch (torch.Tensor): [batch_size, n_ants]
            current_i (int): number of current batch.
            nominal_batch_size (int): a_scoring_batch_size from config.toml
        Returns:
            torch.Tensor [batch_size, n_ants + 1]
                anaphoricity scores for the pairs + a dummy column
        """
        device = top_indices_batch.device
        batch_size = top_indices_batch.size(0)
        n_ants = top_indices_batch.size(1)
        # [batch_size, n_ants, pair_emb]
        pair_matrix = self._get_pair_matrix(
           all_mentions, mentions_batch, pw_batch, top_indices_batch)
        # [n_nodes, mention_emb]
        x = all_mentions

        flattened = torch.flatten(top_indices_batch)
        # [1, batch_size * n_ants]
        flattened = torch.unsqueeze(flattened, 0)
        idx = torch.zeros([1, batch_size*n_ants], device=device)
        for i in range(batch_size):
            idx[0, i*n_ants:(i+1)*n_ants] = torch.ones(n_ants, device=device)*(i + current_i)
        # [2, batch_size * n_ants]
        edge_index = torch.cat([flattened, idx], dim=0)
        # [2, 2 * batch_size * n_ants]
        edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1).long()

        # [batch_size * n_ants, 60]
        pw_batch_tmp = pw_batch.reshape(-1, pw_batch.size(-1))
        # [batch_size * n_ants, 60]
        edge_features = torch.cat([pw_batch_tmp, pw_batch_tmp], dim=0)

        # [batch_size, n_ants]
        scores = top_rough_scores_batch + self.gnn(x, edge_index, edge_features, batch_size)
        scores = utils.add_dummy(scores, eps=True)

        return scores

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.out(self.hidden(x))
        return x.squeeze(2)

    @staticmethod
    def _get_pair_matrix(all_mentions: torch.Tensor,
                         mentions_batch: torch.Tensor,
                         pw_batch: torch.Tensor,
                         top_indices_batch: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb],
                all the valid mentions of the document,
                can be on a different device
            mentions_batch (torch.Tensor): [batch_size, mention_emb],
                the mentions of the current batch,
                is expected to be on the current device
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb],
                pairwise features of the current batch,
                is expected to be on the current device
            top_indices_batch (torch.Tensor): [batch_size, n_ants],
                indices of antecedents of each mention

        Returns:
            torch.Tensor: [batch_size, n_ants, pair_emb]
        """
        emb_size = mentions_batch.shape[1]
        n_ants = pw_batch.shape[1]

        a_mentions = mentions_batch.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = all_mentions[top_indices_batch]
        similarity = a_mentions * b_mentions

        out = torch.cat((a_mentions, b_mentions, similarity, pw_batch), dim=2)
        return out
