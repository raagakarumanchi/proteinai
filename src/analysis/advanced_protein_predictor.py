import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from functools import lru_cache
import json
import os
from pathlib import Path
import re

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AdvancedProteinDataset(Dataset):
    """Advanced dataset for multi-modal protein analysis"""
    
    def __init__(self, sequences, structures=None, functions=None, labels=None, graphs=None):
        self.sequences = sequences
        self.structures = structures
        self.functions = functions
        self.labels = labels
        self.graphs = graphs
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {'sequence': self.sequences[idx]}
        if self.structures is not None:
            item['structure'] = self.structures[idx]
        if self.functions is not None:
            item['function'] = self.functions[idx]
        if self.labels is not None:
            item['label'] = self.labels[idx]
        if self.graphs is not None:
            item['graph'] = self.graphs[idx]
        return item


class ESM2Encoder(nn.Module):
    """ESM-2 based protein sequence encoder"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze ESM-2 parameters for efficiency
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.embedding_dim = self.model.config.hidden_size
        
    def forward(self, sequences):
        """Encode protein sequences using ESM-2"""
        # Tokenize sequences
        encoded = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # Get sequence representations (mean pooling)
        embeddings = outputs.last_hidden_state
        attention_mask = encoded['attention_mask']
        
        # Masked mean pooling
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        pooled_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        return pooled_embeddings, embeddings, attention_mask


class GraphAttentionNetwork(nn.Module):
    """Graph Neural Network for protein structure analysis"""
    
    def __init__(self, input_dim=128, hidden_dim=256, num_heads=8, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.1))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.1))
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass through GNN"""
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
        
        x = self.final_proj(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class FunctionAwareAttention(nn.Module):
    """Function-specific attention mechanism"""
    
    def __init__(self, input_dim=512, num_functions=10, num_heads=8):
        super().__init__()
        self.num_functions = num_functions
        self.input_dim = input_dim
        
        # Function-specific query projections
        self.function_queries = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_functions)
        ])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        
        # Function-specific value projections
        self.function_values = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_functions)
        ])
        
    def forward(self, x, function_id=None):
        """Apply function-specific attention"""
        batch_size, seq_len, _ = x.shape
        
        if function_id is not None:
            # Use specific function attention
            query = self.function_queries[function_id](x)
            value = self.function_values[function_id](x)
            
            attended, _ = self.attention(query, x, value)
        else:
            # Use all function attentions and combine
            attended_outputs = []
            for i in range(self.num_functions):
                query = self.function_queries[i](x)
                value = self.function_values[i](x)
                attended, _ = self.attention(query, x, value)
                attended_outputs.append(attended)
            
            # Combine all function-specific attentions
            attended = torch.stack(attended_outputs, dim=1).mean(dim=1)
        
        return attended


class MultiModalFusion(nn.Module):
    """Multi-modal fusion layer"""
    
    def __init__(self, seq_dim=320, graph_dim=256, hidden_dim=512):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, seq_features, graph_features):
        """Fuse sequence and graph features"""
        # Project to same dimension
        seq_proj = self.seq_proj(seq_features)
        graph_proj = self.graph_proj(graph_features)
        
        # Cross-modal attention
        seq_attended, _ = self.cross_attention(seq_proj, graph_proj, graph_proj)
        graph_attended, _ = self.cross_attention(graph_proj, seq_proj, seq_proj)
        
        # Concatenate and fuse
        fused = torch.cat([seq_attended, graph_attended], dim=-1)
        fused = self.fusion(fused)
        
        return fused


class AdvancedProteinFunctionPredictor(nn.Module):
    """Advanced multi-modal protein function predictor"""
    
    def __init__(self, num_functions=10, num_active_sites=5):
        super().__init__()
        
        # Phase 1: ESM-2 encoder
        self.esm_encoder = ESM2Encoder()
        seq_dim = self.esm_encoder.embedding_dim
        
        # Phase 2: Graph Neural Network
        self.gnn = GraphAttentionNetwork(input_dim=128, hidden_dim=256)
        
        # Phase 3: Multi-modal fusion
        self.fusion = MultiModalFusion(seq_dim=seq_dim, graph_dim=256)
        
        # Function-specific attention
        self.function_attention = FunctionAwareAttention(input_dim=512, num_functions=num_functions)
        
        # Multi-task heads
        self.function_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_functions)
        )
        
        self.active_site_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.stability_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Hierarchical function prediction
        self.broad_classifier = nn.Linear(512, 4)  # Enzyme, Receptor, Structural, Other
        self.specific_classifier = nn.Linear(512, num_functions)
        
    def forward(self, sequences, graphs=None, function_id=None):
        """Forward pass through the complete model"""
        
        # Phase 1: ESM-2 encoding
        seq_embeddings, seq_features, attention_mask = self.esm_encoder(sequences)
        
        # Phase 2: Graph processing (if available)
        if graphs is not None:
            graph_embeddings = self.gnn(graphs.x, graphs.edge_index, graphs.batch)
        else:
            # Create dummy graph features
            batch_size = seq_embeddings.size(0)
            graph_embeddings = torch.zeros(batch_size, 256, device=seq_embeddings.device)
        
        # Phase 3: Multi-modal fusion
        fused_features = self.fusion(seq_embeddings, graph_embeddings)
        
        # Function-specific attention
        attended_features = self.function_attention(fused_features.unsqueeze(1), function_id)
        attended_features = attended_features.squeeze(1)
        
        # Multi-task predictions
        function_logits = self.function_classifier(attended_features)
        active_site_logits = self.active_site_predictor(attended_features)
        stability_logits = self.stability_predictor(attended_features)
        
        # Hierarchical predictions
        broad_logits = self.broad_classifier(attended_features)
        specific_logits = self.specific_classifier(attended_features)
        
        return {
            'function_logits': function_logits,
            'active_site_logits': active_site_logits,
            'stability_logits': stability_logits,
            'broad_logits': broad_logits,
            'specific_logits': specific_logits,
            'attended_features': attended_features,
            'seq_features': seq_features,
            'attention_mask': attention_mask
        }


class AdvancedProteinPredictorAI:
    """Complete AI system for advanced protein analysis"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # Protein function categories
        self.function_categories = {
            0: 'Enzyme',
            1: 'Receptor', 
            2: 'Transporter',
            3: 'Structural',
            4: 'Transcription Factor',
            5: 'Kinase',
            6: 'Phosphatase',
            7: 'Hydrolase',
            8: 'Oxidoreductase',
            9: 'Other'
        }
        
        self.broad_categories = {
            0: 'Enzyme',
            1: 'Receptor',
            2: 'Structural', 
            3: 'Other'
        }
        
        # Initialize model
        self.model = AdvancedProteinFunctionPredictor(
            num_functions=len(self.function_categories)
        ).to(self.device)
        
        # Load pretrained weights if available
        self.load_pretrained_model()
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info("Advanced Protein Function Predictor initialized")
    
    def load_pretrained_model(self):
        """Load pretrained model weights"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "advanced_protein_model.pth"
        if model_path.exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pretrained advanced protein model")
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {e}")
    
    def save_model(self):
        """Save trained model"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        torch.save(self.model.state_dict(), model_dir / "advanced_protein_model.pth")
        logger.info("Advanced protein model saved")
    
    def create_protein_graph(self, sequence: str, structure: str = None) -> Data:
        """Create graph representation of protein"""
        n_nodes = len(sequence)
        
        # Node features (amino acid properties)
        node_features = []
        for aa in sequence:
            features = self._get_aa_features(aa)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices (connect nearby residues)
        edge_index = []
        for i in range(n_nodes):
            for j in range(max(0, i-2), min(n_nodes, i+3)):
                if i != j:
                    edge_index.append([i, j])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index)
    
    def _get_aa_features(self, aa: str) -> List[float]:
        """Get amino acid features"""
        # Hydrophobicity, charge, size, etc.
        features = {
            'A': [1.8, 0, 89.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'R': [-4.5, 1, 174.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'N': [-3.5, 0, 132.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'D': [-3.5, -1, 133.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'C': [2.5, 0, 121.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Q': [-3.5, 0, 146.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'E': [-3.5, -1, 147.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'G': [-0.4, 0, 75.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'H': [-3.2, 1, 155.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'I': [4.5, 0, 131.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'L': [3.8, 0, 131.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'K': [-3.9, 1, 146.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'M': [1.9, 0, 149.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'F': [2.8, 0, 165.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'P': [-1.6, 0, 115.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'S': [-0.8, 0, 105.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'T': [-0.7, 0, 119.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'W': [-0.9, 0, 204.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Y': [-1.3, 0, 181.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'V': [4.2, 0, 117.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        return features.get(aa.upper(), [0] * 21)
    
    def predict_comprehensive(self, sequence: str, structure: str = None) -> Dict:
        """Comprehensive protein analysis"""
        if not sequence:
            return {'error': 'No sequence provided'}
        
        with torch.no_grad():
            # Create graph representation
            graph = self.create_protein_graph(sequence, structure)
            graph = graph.to(self.device)
            
            # Get predictions
            outputs = self.model([sequence], graphs=graph)
            
            # Process predictions
            function_probs = F.softmax(outputs['function_logits'], dim=-1)
            broad_probs = F.softmax(outputs['broad_logits'], dim=-1)
            active_site_prob = torch.sigmoid(outputs['active_site_logits'])
            stability_prob = torch.sigmoid(outputs['stability_logits'])
            
            # Get top predictions
            top_function = torch.argmax(function_probs, dim=-1).item()
            top_broad = torch.argmax(broad_probs, dim=-1).item()
            
            # Get all function probabilities
            all_functions = {}
            for i, prob in enumerate(function_probs[0]):
                all_functions[self.function_categories[i]] = prob.item()
            
            all_broad = {}
            for i, prob in enumerate(broad_probs[0]):
                all_broad[self.broad_categories[i]] = prob.item()
            
            return {
                'sequence': sequence,
                'length': len(sequence),
                'predicted_function': self.function_categories[top_function],
                'function_confidence': function_probs[0][top_function].item(),
                'broad_category': self.broad_categories[top_broad],
                'broad_confidence': broad_probs[0][top_broad].item(),
                'all_functions': all_functions,
                'all_broad_categories': all_broad,
                'active_site_probability': active_site_prob[0].item(),
                'stability_score': stability_prob[0].item(),
                'attention_weights': outputs['attended_features'][0].cpu().numpy(),
                'seq_features': outputs['seq_features'][0].cpu().numpy(),
                'attention_mask': outputs['attention_mask'][0].cpu().numpy()
            }
    
    def analyze_functional_regions(self, sequence: str, structure: str = None) -> Dict:
        """Analyze functional regions using attention weights"""
        prediction = self.predict_comprehensive(sequence, structure)
        
        if 'error' in prediction:
            return prediction
        
        attention_weights = prediction['attention_weights']
        seq_features = prediction['seq_features']
        
        # Find high-attention regions
        if len(attention_weights.shape) > 1:
            avg_attention = np.mean(attention_weights, axis=1)
        else:
            avg_attention = attention_weights
        
        threshold = np.percentile(avg_attention, 85)
        
        functional_regions = []
        in_region = False
        start = 0
        
        for i, attention in enumerate(avg_attention):
            if attention > threshold and not in_region:
                start = i
                in_region = True
            elif attention <= threshold and in_region:
                if i - start >= 5:
                    functional_regions.append({
                        'start': start,
                        'end': i - 1,
                        'sequence': sequence[start:i],
                        'attention_score': np.mean(avg_attention[start:i]),
                        'importance': 'High' if np.mean(avg_attention[start:i]) > np.percentile(avg_attention, 95) else 'Medium'
                    })
                in_region = False
        
        # Handle final region
        if in_region and len(sequence) - start >= 5:
            functional_regions.append({
                'start': start,
                'end': len(sequence) - 1,
                'sequence': sequence[start:],
                'attention_score': np.mean(avg_attention[start:]),
                'importance': 'High' if np.mean(avg_attention[start:]) > np.percentile(avg_attention, 95) else 'Medium'
            })
        
        return {
            'functional_regions': functional_regions,
            'total_regions': len(functional_regions),
            'high_importance_regions': len([r for r in functional_regions if r['importance'] == 'High']),
            'attention_distribution': {
                'mean': np.mean(avg_attention),
                'std': np.std(avg_attention),
                'max': np.max(avg_attention),
                'min': np.min(avg_attention)
            }
        }
    
    def train_model(self, sequences: List[str], functions: List[str], epochs: int = 100):
        """Train the advanced model"""
        logger.info("Starting advanced protein function model training...")
        
        # Create graphs for all sequences
        graphs = [self.create_protein_graph(seq) for seq in sequences]
        
        # Create dataset
        dataset = AdvancedProteinDataset(sequences, functions=functions, graphs=graphs)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                sequences_batch = batch['sequence']
                functions_batch = batch['function']
                graphs_batch = Batch.from_data_list(batch['graph']).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(sequences_batch, graphs_batch)
                
                # Multi-task loss
                function_loss = criterion(outputs['function_logits'], functions_batch)
                broad_loss = criterion(outputs['broad_logits'], functions_batch // 3)  # Simplified broad categories
                
                total_loss_batch = function_loss + 0.5 * broad_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                _, predicted = torch.max(outputs['function_logits'], 1)
                total += functions_batch.size(0)
                correct += (predicted == functions_batch).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.model.eval()
        self.save_model()
        logger.info("Advanced protein function model training completed")
