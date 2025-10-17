import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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


class ProteinFunctionDataset(Dataset):
    """Dataset for protein sequences and their functions"""
    
    def __init__(self, sequences, functions=None, labels=None):
        self.sequences = sequences
        self.functions = functions
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {'sequence': self.sequences[idx]}
        if self.functions is not None:
            item['function'] = self.functions[idx]
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item


class ProteinFunctionEmbedding(nn.Module):
    """Advanced protein sequence embedding for function prediction"""
    
    def __init__(self, vocab_size=21, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(2000, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Attention pooling for global representation
        pooled, _ = self.attention_pool(x, x, x)
        global_repr = pooled.mean(dim=1)
        
        return x, global_repr


class ProteinFunctionPredictor(nn.Module):
    """Deep learning model for protein function prediction"""
    
    def __init__(self, vocab_size=21, embed_dim=256, num_classes=10):
        super().__init__()
        self.embedding = ProteinFunctionEmbedding(vocab_size, embed_dim)
        
        # CNN layers for local functional patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        # LSTM for sequence dependencies
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Function-specific attention
        self.function_attention = nn.MultiheadAttention(128, 8, batch_first=True)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 512),  # Global + LSTM features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, mask=None):
        # Get embeddings
        seq_embeddings, global_repr = self.embedding(x, mask)
        
        # CNN processing
        cnn_input = seq_embeddings.transpose(1, 2)  # (batch, embed_dim, seq_len)
        cnn_output = self.conv_layers(cnn_input)
        cnn_output = cnn_output.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM processing
        lstm_output, _ = self.lstm(cnn_output)
        
        # Function-specific attention
        attended_output, _ = self.function_attention(lstm_output, lstm_output, lstm_output)
        lstm_global = attended_output.mean(dim=1)
        
        # Combine global and LSTM features
        combined_features = torch.cat([global_repr, lstm_global], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits, seq_embeddings, attended_output


class ProteinFunctionPredictorAI:
    """AI system for predicting protein functions"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        
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
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Initialize model
        self.model = ProteinFunctionPredictor(
            vocab_size=21,
            embed_dim=256,
            num_classes=len(self.function_categories)
        ).to(self.device)
        
        # Load pretrained weights if available
        self.load_pretrained_model()
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info("Protein Function Predictor initialized")
    
    def load_pretrained_model(self):
        """Load pretrained model weights"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "protein_function_model.pth"
        if model_path.exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pretrained protein function model")
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {e}")
    
    def save_model(self):
        """Save trained model"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        torch.save(self.model.state_dict(), model_dir / "protein_function_model.pth")
        logger.info("Protein function model saved")
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to tensor"""
        encoded = [self.aa_to_idx.get(aa, 0) for aa in sequence.upper()]
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict_function(self, sequence: str) -> Dict:
        """Predict protein function from sequence"""
        if not sequence:
            return {'function': 'Unknown', 'confidence': 0.0, 'all_predictions': {}}
        
        with torch.no_grad():
            x = self.encode_sequence(sequence)
            logits, embeddings, attention = self.model(x)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
            
            # Get predictions
            pred_class = predicted_class[0].item()
            confidence = probabilities[0][pred_class].item()
            
            # Get all function probabilities
            all_predictions = {}
            for i, prob in enumerate(probabilities[0]):
                all_predictions[self.function_categories[i]] = prob.item()
            
            return {
                'function': self.function_categories[pred_class],
                'confidence': confidence,
                'all_predictions': all_predictions,
                'sequence_length': len(sequence),
                'embeddings': embeddings[0].cpu().numpy(),
                'attention_weights': attention[0].cpu().numpy()
            }
    
    def predict_multiple_functions(self, sequences: List[str]) -> List[Dict]:
        """Predict functions for multiple sequences"""
        results = []
        for seq in sequences:
            result = self.predict_function(seq)
            results.append(result)
        return results
    
    def analyze_functional_regions(self, sequence: str) -> Dict:
        """Analyze functional regions in protein sequence"""
        if not sequence:
            return {'functional_regions': [], 'active_sites': []}
        
        prediction = self.predict_function(sequence)
        attention_weights = prediction['attention_weights']
        
        # Find high-attention regions (likely functional sites)
        avg_attention = np.mean(attention_weights, axis=1)
        threshold = np.percentile(avg_attention, 80)
        
        functional_regions = []
        active_sites = []
        
        # Identify functional regions
        in_region = False
        start = 0
        
        for i, attention in enumerate(avg_attention):
            if attention > threshold and not in_region:
                start = i
                in_region = True
            elif attention <= threshold and in_region:
                if i - start >= 5:  # Minimum region length
                    functional_regions.append({
                        'start': start,
                        'end': i - 1,
                        'sequence': sequence[start:i],
                        'attention_score': np.mean(avg_attention[start:i])
                    })
                in_region = False
        
        # Handle case where sequence ends in functional region
        if in_region and len(sequence) - start >= 5:
            functional_regions.append({
                'start': start,
                'end': len(sequence) - 1,
                'sequence': sequence[start:],
                'attention_score': np.mean(avg_attention[start:])
            })
        
        # Predict active sites based on common patterns
        active_sites = self._predict_active_sites(sequence, prediction['function'])
        
        return {
            'functional_regions': functional_regions,
            'active_sites': active_sites,
            'total_functional_regions': len(functional_regions)
        }
    
    def _predict_active_sites(self, sequence: str, predicted_function: str) -> List[Dict]:
        """Predict active sites based on function and sequence patterns"""
        active_sites = []
        
        # Common active site patterns
        patterns = {
            'Enzyme': ['GXGXXG', 'DXXH', 'HXXE', 'SXXXK'],
            'Kinase': ['GXGXXG', 'DFG', 'APE'],
            'Phosphatase': ['CX5R', 'DX2H'],
            'Hydrolase': ['GXSXG', 'DX2H'],
            'Oxidoreductase': ['GXGXXG', 'FAD', 'NAD']
        }
        
        if predicted_function in patterns:
            for pattern in patterns[predicted_function]:
                for match in re.finditer(pattern, sequence):
                    active_sites.append({
                        'start': match.start(),
                        'end': match.end() - 1,
                        'sequence': match.group(),
                        'pattern': pattern,
                        'type': 'Active Site'
                    })
        
        return active_sites
    
    def get_function_explanation(self, function: str) -> Dict:
        """Get explanation of predicted protein function"""
        explanations = {
            'Enzyme': {
                'description': 'Catalyzes biochemical reactions',
                'examples': 'DNA polymerase, RNA polymerase, proteases',
                'common_patterns': 'Active sites with catalytic residues',
                'applications': 'Drug targets, industrial biocatalysis'
            },
            'Receptor': {
                'description': 'Binds to specific molecules (ligands)',
                'examples': 'GPCRs, ion channels, hormone receptors',
                'common_patterns': 'Transmembrane domains, binding pockets',
                'applications': 'Drug discovery, signal transduction'
            },
            'Transporter': {
                'description': 'Moves molecules across cell membranes',
                'examples': 'Glucose transporters, ion pumps, ABC transporters',
                'common_patterns': 'Multiple transmembrane domains',
                'applications': 'Drug delivery, nutrient uptake'
            },
            'Structural': {
                'description': 'Provides structural support to cells',
                'examples': 'Collagen, actin, tubulin, keratin',
                'common_patterns': 'Repetitive sequences, coiled-coil domains',
                'applications': 'Tissue engineering, biomaterials'
            },
            'Transcription Factor': {
                'description': 'Regulates gene expression',
                'examples': 'p53, NF-κB, STAT proteins',
                'common_patterns': 'DNA-binding domains, activation domains',
                'applications': 'Gene therapy, cancer treatment'
            },
            'Kinase': {
                'description': 'Adds phosphate groups to proteins',
                'examples': 'Protein kinase A, MAP kinases, tyrosine kinases',
                'common_patterns': 'ATP-binding site, catalytic loop',
                'applications': 'Cancer therapy, signal transduction'
            },
            'Phosphatase': {
                'description': 'Removes phosphate groups from proteins',
                'examples': 'Protein phosphatase 1, PTP1B, calcineurin',
                'common_patterns': 'Catalytic cysteine, metal binding',
                'applications': 'Diabetes treatment, immune regulation'
            },
            'Hydrolase': {
                'description': 'Breaks down molecules using water',
                'examples': 'Proteases, lipases, nucleases, esterases',
                'common_patterns': 'Catalytic triad, oxyanion hole',
                'applications': 'Digestion, cellular cleanup, biotechnology'
            },
            'Oxidoreductase': {
                'description': 'Catalyzes oxidation-reduction reactions',
                'examples': 'Cytochrome P450, alcohol dehydrogenase, catalase',
                'common_patterns': 'Cofactor binding sites, redox centers',
                'applications': 'Drug metabolism, energy production'
            },
            'Other': {
                'description': 'Miscellaneous or unknown function',
                'examples': 'Various proteins with diverse roles',
                'common_patterns': 'Variable, depends on specific protein',
                'applications': 'Research target, potential drug discovery'
            }
        }
        
        return explanations.get(function, explanations['Other'])
    
    def train_model(self, sequences: List[str], functions: List[str], epochs: int = 50):
        """Train the protein function prediction model"""
        logger.info("Starting protein function model training...")
        
        # Encode sequences and functions
        encoded_sequences = []
        encoded_functions = []
        
        for seq, func in zip(sequences, functions):
            if len(seq) > 10:  # Minimum sequence length
                encoded_sequences.append([self.aa_to_idx.get(aa, 0) for aa in seq.upper()])
                encoded_functions.append(func)
        
        # Encode function labels
        encoded_labels = self.label_encoder.fit_transform(encoded_functions)
        
        # Create dataset and dataloader
        dataset = ProteinFunctionDataset(encoded_sequences, encoded_functions, encoded_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                sequences = torch.tensor(batch['sequence']).to(self.device)
                labels = torch.tensor(batch['label']).to(self.device)
                
                optimizer.zero_grad()
                logits, _, _ = self.model(sequences)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.model.eval()
        self.save_model()
        logger.info("Protein function model training completed")
    
    def benchmark_model(self, test_sequences: List[str], test_functions: List[str]) -> Dict:
        """Benchmark model performance"""
        if not test_sequences or not test_functions:
            return {'accuracy': 0, 'details': {}}
        
        predictions = self.predict_multiple_functions(test_sequences)
        correct = 0
        total = len(test_sequences)
        
        for pred, true_func in zip(predictions, test_functions):
            if pred['function'] == true_func:
                correct += 1
        
        accuracy = correct / total
        
        # Calculate per-function accuracy
        function_accuracies = {}
        for func in set(test_functions):
            func_correct = 0
            func_total = 0
            for pred, true_func in zip(predictions, test_functions):
                if true_func == func:
                    func_total += 1
                    if pred['function'] == func:
                        func_correct += 1
            function_accuracies[func] = func_correct / func_total if func_total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'function_accuracies': function_accuracies,
            'average_confidence': np.mean([p['confidence'] for p in predictions])
        }
