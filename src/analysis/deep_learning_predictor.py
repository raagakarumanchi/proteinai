import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from functools import lru_cache
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    """Dataset for protein sequences and structures"""
    
    def __init__(self, sequences, structures=None, labels=None):
        self.sequences = sequences
        self.structures = structures
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {'sequence': self.sequences[idx]}
        if self.structures is not None:
            item['structure'] = self.structures[idx]
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item


class ProteinEmbedding(nn.Module):
    """Protein sequence embedding using transformer architecture"""
    
    def __init__(self, vocab_size=21, embed_dim=128, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        return x


class SecondaryStructurePredictor(nn.Module):
    """Deep learning model for secondary structure prediction"""
    
    def __init__(self, vocab_size=21, embed_dim=128, num_classes=3):
        super().__init__()
        self.embedding = ProteinEmbedding(vocab_size, embed_dim)
        
        # CNN layers for local patterns
        self.conv1d = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # LSTM for sequence dependencies
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, mask=None):
        # Get embeddings
        x = self.embedding(x, mask)
        
        # CNN processing
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class StabilityPredictor(nn.Module):
    """Deep learning model for protein stability prediction"""
    
    def __init__(self, vocab_size=21, embed_dim=128):
        super().__init__()
        self.embedding = ProteinEmbedding(vocab_size, embed_dim)
        
        # Global pooling and regression
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, mask=None):
        x = self.embedding(x, mask)
        
        # Global average pooling
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)
        
        x = self.regressor(x)
        return x.squeeze(-1)


class ContactMapPredictor(nn.Module):
    """Deep learning model for contact map prediction"""
    
    def __init__(self, vocab_size=21, embed_dim=128):
        super().__init__()
        self.embedding = ProteinEmbedding(vocab_size, embed_dim)
        
        # Pairwise interaction network
        self.interaction_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Get embeddings
        x = self.embedding(x, mask)
        
        # Create pairwise features
        x1 = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x2 = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Concatenate pairwise features
        pairwise = torch.cat([x1, x2], dim=-1)
        
        # Predict contacts
        contacts = self.interaction_net(pairwise).squeeze(-1)
        
        return contacts


class DeepLearningPredictor:
    """Deep learning-based protein structure predictor"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        self.ss_to_idx = {'H': 0, 'E': 1, 'C': 2}
        self.idx_to_ss = {0: 'H', 1: 'E', 2: 'C'}
        
        # Initialize models
        self.ss_model = SecondaryStructurePredictor().to(self.device)
        self.stability_model = StabilityPredictor().to(self.device)
        self.contact_model = ContactMapPredictor().to(self.device)
        
        # Load pretrained weights if available
        self.load_pretrained_models()
        
        # Set to evaluation mode
        self.ss_model.eval()
        self.stability_model.eval()
        self.contact_model.eval()
        
        logger.info("Deep learning models initialized")
    
    def load_pretrained_models(self):
        """Load pretrained model weights"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # Try to load pretrained weights
        for model_name, model in [('ss', self.ss_model), ('stability', self.stability_model), ('contact', self.contact_model)]:
            model_path = model_dir / f"{model_name}_model.pth"
            if model_path.exists():
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info(f"Loaded pretrained {model_name} model")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} model: {e}")
    
    def save_models(self):
        """Save trained models"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        torch.save(self.ss_model.state_dict(), model_dir / "ss_model.pth")
        torch.save(self.stability_model.state_dict(), model_dir / "stability_model.pth")
        torch.save(self.contact_model.state_dict(), model_dir / "contact_model.pth")
        
        logger.info("Models saved")
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to tensor"""
        encoded = [self.aa_to_idx.get(aa, 0) for aa in sequence.upper()]
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict_secondary_structure(self, sequence: str) -> Dict:
        """Predict secondary structure using deep learning"""
        if not sequence:
            return {'structure': '', 'confidence': [], 'regions': {'helix': [], 'sheet': [], 'coil': []}}
        
        with torch.no_grad():
            x = self.encode_sequence(sequence)
            logits = self.ss_model(x)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to structure string
            structure = ''.join([self.idx_to_ss[pred.item()] for pred in predictions[0]])
            
            # Get confidence scores
            confidence = probabilities[0].cpu().numpy()
            
            # Find regions
            regions = self._find_structure_regions(structure)
            
            return {
                'structure': structure,
                'confidence': confidence.tolist(),
                'regions': regions,
                'helix_content': structure.count('H') / len(structure),
                'sheet_content': structure.count('E') / len(structure),
                'coil_content': structure.count('C') / len(structure)
            }
    
    def predict_stability(self, sequence: str) -> Dict:
        """Predict protein stability using deep learning"""
        if not sequence:
            return {'stability_score': 0.0, 'confidence': 0.0}
        
        with torch.no_grad():
            x = self.encode_sequence(sequence)
            stability = self.stability_model(x)
            stability_score = torch.sigmoid(stability).item()
            
            return {
                'stability_score': stability_score,
                'confidence': 0.9,  # Placeholder
                'features': self._extract_stability_features(sequence)
            }
    
    def predict_contact_map(self, sequence: str) -> np.ndarray:
        """Predict contact map using deep learning"""
        if not sequence or len(sequence) < 2:
            return np.array([])
        
        with torch.no_grad():
            x = self.encode_sequence(sequence)
            contacts = self.contact_model(x)
            contact_map = torch.sigmoid(contacts[0]).cpu().numpy()
            
            return contact_map
    
    def predict_aggregation(self, sequence: str) -> Dict:
        """Predict aggregation propensity using deep learning"""
        if not sequence:
            return {'aggregation_score': 0.0, 'hotspots': []}
        
        # Use stability model as proxy for aggregation
        stability_result = self.predict_stability(sequence)
        
        # Invert stability for aggregation (simplified)
        aggregation_score = 1.0 - stability_result['stability_score']
        
        # Find aggregation hotspots (simplified)
        hotspots = self._find_aggregation_hotspots(sequence, aggregation_score)
        
        return {
            'aggregation_score': aggregation_score,
            'hotspots': hotspots,
            'confidence': 0.8
        }
    
    def _find_structure_regions(self, structure: str) -> Dict:
        """Find continuous regions of secondary structure"""
        regions = {'helix': [], 'sheet': [], 'coil': []}
        
        current_ss = structure[0]
        start = 0
        
        for i in range(1, len(structure)):
            if structure[i] != current_ss:
                if current_ss in regions:
                    regions[current_ss].append((start, i-1))
                current_ss = structure[i]
                start = i
        
        # Add final region
        if current_ss in regions:
            regions[current_ss].append((start, len(structure)-1))
        
        return regions
    
    def _extract_stability_features(self, sequence: str) -> Dict:
        """Extract features for stability analysis"""
        # Simple feature extraction
        features = {
            'length': len(sequence),
            'hydrophobic_ratio': sum(1 for aa in sequence if aa in 'AILMFPWYV') / len(sequence),
            'charged_ratio': sum(1 for aa in sequence if aa in 'DEKR') / len(sequence),
            'aromatic_ratio': sum(1 for aa in sequence if aa in 'FWY') / len(sequence),
            'proline_ratio': sequence.count('P') / len(sequence),
            'glycine_ratio': sequence.count('G') / len(sequence)
        }
        return features
    
    def _find_aggregation_hotspots(self, sequence: str, aggregation_score: float) -> List[Dict]:
        """Find potential aggregation hotspots"""
        hotspots = []
        window_size = 6
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            
            # Simple hotspot scoring
            hydrophobic_count = sum(1 for aa in window if aa in 'AILMFPWYV')
            beta_sheet_propensity = sum(1 for aa in window if aa in 'FILVWY')
            
            hotspot_score = (hydrophobic_count + beta_sheet_propensity) / (window_size * 2)
            
            if hotspot_score > 0.6:
                hotspots.append({
                    'start': i,
                    'end': i + window_size - 1,
                    'sequence': window,
                    'score': hotspot_score
                })
        
        return hotspots
    
    def train_models(self, sequences: List[str], structures: List[str] = None, 
                    stabilities: List[float] = None, epochs: int = 10):
        """Train the deep learning models"""
        logger.info("Starting model training...")
        
        # Prepare data
        if structures:
            self._train_secondary_structure(sequences, structures, epochs)
        
        if stabilities:
            self._train_stability(sequences, stabilities, epochs)
        
        # Save trained models
        self.save_models()
        logger.info("Training completed")
    
    def _train_secondary_structure(self, sequences: List[str], structures: List[str], epochs: int):
        """Train secondary structure prediction model"""
        # Encode sequences and structures
        encoded_sequences = []
        encoded_structures = []
        
        for seq, struct in zip(sequences, structures):
            if len(seq) == len(struct):
                encoded_sequences.append([self.aa_to_idx.get(aa, 0) for aa in seq.upper()])
                encoded_structures.append([self.ss_to_idx.get(ss, 2) for ss in struct])
        
        # Create dataset and dataloader
        dataset = ProteinDataset(encoded_sequences, encoded_structures)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(self.ss_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.ss_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                sequences = torch.tensor(batch['sequence']).to(self.device)
                structures = torch.tensor(batch['structure']).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.ss_model(sequences)
                loss = criterion(outputs.view(-1, 3), structures.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.ss_model.eval()
    
    def _train_stability(self, sequences: List[str], stabilities: List[float], epochs: int):
        """Train stability prediction model"""
        # Encode sequences
        encoded_sequences = []
        for seq in sequences:
            encoded_sequences.append([self.aa_to_idx.get(aa, 0) for aa in seq.upper()])
        
        # Create dataset and dataloader
        dataset = ProteinDataset(encoded_sequences, labels=stabilities)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(self.stability_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.stability_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                sequences = torch.tensor(batch['sequence']).to(self.device)
                stabilities = torch.tensor(batch['label'], dtype=torch.float).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.stability_model(sequences)
                loss = criterion(outputs, stabilities)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.stability_model.eval()
