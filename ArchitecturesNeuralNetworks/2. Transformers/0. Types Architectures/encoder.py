"""
Encoder ALGORITHM BY SCRATCH IN PYTORCH, the main ideia and structure:

1. Embedding (embdding + pos embedding)
2. TransformerBlock(
    - Attention (Self attetion + NO MASK IN THE SCORES) + residual connection and normalization
    - FeedForwardNetwork(Normal dense network, with non-linearity) + residual connection and normalization
)
3. Final Dense (The final layer to predict the num_classes)

"""


import torch 
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


# Global Variables
VOCAB_SIZE = 50257
MAX_SEQ_LEN = 512
D_MODEL = 768
NHEADS = 12
NBLOCKS = 12
NUM_CLASSES = 1

class BERTConfig(BaseModel):
    """
        Class BERT: Configure the right values for each args

    """

    vocab_size: int = Field(..., gt=0)
    max_seq_len: int = Field(..., gt=0)
    d_model: int = Field(..., gt=0)
    nheads: int = Field(..., gt=0)
    nblocks: int = Field(..., gt=0)
    num_classes: int = Field(..., gt=0)

config = BERTConfig(
    vocab_size = VOCAB_SIZE,
    max_seq_len = MAX_SEQ_LEN,
    d_model = D_MODEL,
    nheads = NHEADS,
    nblocks = NBLOCKS,
    num_classes = NUM_CLASSES
)


class MultiHeadAttention(nn.Module):

    """
        Class MultiHeadAttention:
            Self multi head attetion algorithm from scratch with no masked (ENCODER)

        Args:
            d_model = number of dimensions that each token gonna be
            nheads = number of heads that each block have
            d_k = The number of dimensions in each head

        Input:
            The input gonna be an (BATCH, TOKENS, D_MODEL) coming from embedding
            B represents the Batches
            T represents the tokens
            C represents the d_model

        Output:
            The output gonna be an (BATCH, TOKENS, D_MODEL) too
    """

    def __init__(self, d_model: int, nheads: int) -> None: 
        super().__init__()

        self.wQ = nn.Linear(d_model, d_model, bias = False)
        self.wK = nn.Linear(d_model, d_model, bias = False)
        self.wV = nn.Linear(d_model, d_model, bias = False)
        self.wO = nn.Linear(d_model, d_model, bias = False)

        self.nheads = nheads
        self.d_model = d_model
        self.d_k = d_model // nheads


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, t, c = x.shape

        Q = self.wQ(x)
        K = self.wK(x)
        V = self.wV(x)


        Q = Q.reshape(b, t, self.nheads, self.d_k).transpose(1, 2) # (B, NHEADS, T, D_K)
        K = K.reshape(b, t, self.nheads, self.d_k).transpose(1, 2) # (B, NHEADS, T, D_K)
        V = V.reshape(b, t, self.nheads, self.d_k).transpose(1, 2) # (B, NHEADS, T, D_K)


        scores = F.scaled_dot_product_attention(
            query = Q,
            key = K,
            value = V,
            is_causal = False,
            dropout_p = 0.15
        )

        """
        If you want more control and do all from scratch you can do the code:

        scores = (Q @ K.tranpose(-1, -2)) / math.sqrt(d_k) # Here calculate the scores and normalize them
        scores = F.softmax(scores, dim = 1) # Applie the softmax for each token
        scores = scores @ V # Get the new values of the embedding

        
        """


        scores = scores.transpose(1, 2).reshape(b, t, self.d_model) # Return to (B, T, D_MODEL)
        out_proj =  self.wO(scores) # (B, T, D_MODEL)

        return out_proj




class FeedForwardNetwork(nn.Module):

    """
        Class FeedForwardNetwork:
            This part it's post Attetion, here we add the non-linearity, the space gonna be up-projection in 4 times and then applies the non-linearityand down-projection returns in the original shape

        Args:
            d_model = number of dimensions that each token gonna be
            dropout = Value of the dropout

        Input:
            The input gonna be an (BATCH, TOKENS, D_MODEL) coming from Attention

        Output:
            The output gonna be an (BATCH, TOKENS, D_MODEL) too
    """
    
    def __init__(self, d_model: int, dropout: float = 0.15) -> None:
        super().__init__()


        self.ffn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.ffn(x)

        return x



class TransformerBlock(nn.Module):

    """
        Class TransformerBlock:
            Each transformerBlock and their loop, this part it's here we connect the Attention Layer with the FeedForwardNetwork and applie the Norms and Residual Connections

        Args:
            d_model = Number of dimensions that each token gonna be
            nhedads = Number of heads

        Input:
            The input gonna be an (BATCH, TOKENS, D_MODEL) coming from 

        Output:
            The output gonna be an (BATCH, TOKENS, D_MODEL) too
    """

    def __init__(self, d_model: int, nheads: int) -> None:
        super().__init__()

        self.attn = MultiHeadAttention(d_model, nheads)
        self.ffn = FeedForwardNetwork(d_model)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layernorm1(x + self.attn(x))
        x = self.layernorm2(x + self.ffn(x))

        return x



class Encoder(nn.Module):
    """

        Class Encoder (MAIN):
        The main class, control all, this part it's here we connect the:
            - Embedding + TransformersBlock + Logits

        Args:
            vocab_size = Number of tokens (the size of vocabulary)
            max_seq_len = Max value of tokens (the size of context that our model can see)
            d_model = Number of dimensions that each token gonna be
            nhedads = Number of heads
            nblocks = Number of blocks of (TRANSFOMERS BLOCK) that our model gonna have
            num_classes = Number of final classes

        Input:
            The input gonna be an (BATCH, TOKENS)
            B represents the Batches
            T represents the tokens


        Output:
            The output gonna be an (BATCH, NUM_CLASSES)

    """

    def __init__(self, 
                 vocab_size: int, max_seq_len: int,
                 d_model: int, nheads: int, 
                 nblocks: int, num_classes: int
                ) -> None:

    
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nheads) for _ in range (nblocks)
        ])

        self.logits = nn.Linear(d_model, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t =   x.shape
    
            x = self.embedding(x) + self.pos_embedding(torch.arange(t, device = x.device)) # 1. Embedding

            for block in self.blocks: # Attention + Feed Forward, will returbn (B, T, D_MODEL)
                x = block(x) 
 
            x = x[:, 0, :] # (B, D_MODEL)
            logits = self.logits(x) # (B, NUM_CLASSES)

            return logits

model = Encoder(**config.dict())