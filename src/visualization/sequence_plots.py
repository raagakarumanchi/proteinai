import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SequenceVisualizer:
    """Interactive visualizations for biological sequence analysis"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_amino_acid_composition(self, sequences: List[str], labels: List[str] = None) -> go.Figure:
        """Create interactive amino acid composition plot"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        if labels is None:
            labels = [f"Sequence {i+1}" for i in range(len(sequences))]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hydrophobic vs Hydrophilic', 'Charged vs Neutral',
                          'Aromatic vs Aliphatic', 'Overall Composition'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Calculate composition for each sequence
        compositions = []
        for seq in sequences:
            comp = {}
            for aa in amino_acids:
                comp[aa] = seq.count(aa) / len(seq) if seq else 0
            
            # Group properties
            hydrophobic = sum(seq.count(aa) for aa in 'AILMFPWYV') / len(seq) if seq else 0
            hydrophilic = sum(seq.count(aa) for aa in 'NQSTY') / len(seq) if seq else 0
            charged = sum(seq.count(aa) for aa in 'DEKR') / len(seq) if seq else 0
            neutral = sum(seq.count(aa) for aa in 'ACFGILMPSTVWY') / len(seq) if seq else 0
            aromatic = sum(seq.count(aa) for aa in 'FWY') / len(seq) if seq else 0
            aliphatic = sum(seq.count(aa) for aa in 'AILV') / len(seq) if seq else 0
            
            comp.update({
                'hydrophobic': hydrophobic,
                'hydrophilic': hydrophilic,
                'charged': charged,
                'neutral': neutral,
                'aromatic': aromatic,
                'aliphatic': aliphatic
            })
            compositions.append(comp)
        
        # Plot 1: Hydrophobic vs Hydrophilic
        for i, (comp, label) in enumerate(zip(compositions, labels)):
            fig.add_trace(
                go.Scatter(x=[comp['hydrophobic']], y=[comp['hydrophilic']],
                          mode='markers+text', text=[label], textposition="top center",
                          marker=dict(size=10, color=i), name=label),
                row=1, col=1
            )
        
        # Plot 2: Charged vs Neutral
        for i, (comp, label) in enumerate(zip(compositions, labels)):
            fig.add_trace(
                go.Scatter(x=[comp['charged']], y=[comp['neutral']],
                          mode='markers+text', text=[label], textposition="top center",
                          marker=dict(size=10, color=i), name=label, showlegend=False),
                row=1, col=2
            )
        
        # Plot 3: Aromatic vs Aliphatic
        for i, (comp, label) in enumerate(zip(compositions, labels)):
            fig.add_trace(
                go.Scatter(x=[comp['aromatic']], y=[comp['aliphatic']],
                          mode='markers+text', text=[label], textposition="top center",
                          marker=dict(size=10, color=i), name=label, showlegend=False),
                row=2, col=1
            )
        
        # Plot 4: Overall composition (average)
        avg_comp = {aa: np.mean([comp[aa] for comp in compositions]) for aa in amino_acids}
        fig.add_trace(
            go.Bar(x=list(avg_comp.keys()), y=list(avg_comp.values()),
                  name="Average Composition"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Amino Acid Composition Analysis",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_sequence_alignment(self, sequences: List[str], labels: List[str] = None) -> go.Figure:
        """Create interactive sequence alignment visualization"""
        if labels is None:
            labels = [f"Seq {i+1}" for i in range(len(sequences))]
        
        # Create alignment matrix
        max_len = max(len(seq) for seq in sequences)
        alignment_matrix = []
        
        for seq in sequences:
            row = []
            for i in range(max_len):
                if i < len(seq):
                    row.append(seq[i])
                else:
                    row.append('-')
            alignment_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[[ord(c) for c in row] for row in alignment_matrix],
            y=labels,
            colorscale='Viridis',
            showscale=False,
            hovertext=alignment_matrix,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Sequence Alignment Visualization",
            xaxis_title="Position",
            yaxis_title="Sequence",
            height=400
        )
        
        return fig
    
    def plot_evolutionary_tree(self, sequences: List[str], labels: List[str] = None) -> go.Figure:
        """Create phylogenetic tree visualization"""
        if labels is None:
            labels = [f"Species {i+1}" for i in range(len(sequences))]
        
        # Simple distance matrix (for demo purposes)
        n = len(sequences)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate simple Hamming distance
                seq1, seq2 = sequences[i], sequences[j]
                min_len = min(len(seq1), len(seq2))
                if min_len == 0:
                    distance = 1.0
                else:
                    matches = sum(1 for k in range(min_len) if seq1[k] == seq2[k])
                    distance = 1 - (matches / min_len)
                distances[i][j] = distance
                distances[j][i] = distance
        
        # Create dendrogram-like visualization
        fig = go.Figure()
        
        # Simple tree layout (for demo)
        y_positions = np.linspace(0, 1, n)
        
        for i, (label, y_pos) in enumerate(zip(labels, y_positions)):
            fig.add_trace(go.Scatter(
                x=[0], y=[y_pos],
                mode='markers+text',
                text=[label],
                textposition="middle right",
                marker=dict(size=10, color='blue'),
                name=label
            ))
        
        # Add connecting lines (simplified)
        for i in range(n-1):
            fig.add_trace(go.Scatter(
                x=[0, 0.5], y=[y_positions[i], y_positions[i+1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Phylogenetic Tree",
            xaxis=dict(range=[-0.1, 1.1], showticklabels=False),
            yaxis=dict(range=[-0.1, 1.1], showticklabels=False),
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_conservation_analysis(self, sequences: List[str]) -> go.Figure:
        """Plot sequence conservation across positions"""
        if not sequences:
            return go.Figure()
        
        min_length = min(len(seq) for seq in sequences)
        conservation_scores = []
        
        for pos in range(min_length):
            # Get amino acids at this position
            amino_acids = [seq[pos] for seq in sequences if pos < len(seq)]
            
            if not amino_acids:
                conservation_scores.append(0)
                continue
            
            # Calculate conservation (most common amino acid frequency)
            most_common_aa = max(set(amino_acids), key=amino_acids.count)
            conservation = amino_acids.count(most_common_aa) / len(amino_acids)
            conservation_scores.append(conservation)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(min_length)),
            y=conservation_scores,
            mode='lines+markers',
            name='Conservation Score',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        # Add threshold line
        fig.add_hline(y=0.8, line_dash="dash", line_color="gray",
                     annotation_text="High Conservation Threshold")
        
        fig.update_layout(
            title="Sequence Conservation Analysis",
            xaxis_title="Position",
            yaxis_title="Conservation Score",
            height=400
        )
        
        return fig
    
    def plot_3d_structure_prediction(self, sequence: str) -> go.Figure:
        """Create 3D visualization of predicted protein structure"""
        # Simplified 3D structure prediction visualization
        # In reality, this would use tools like AlphaFold or similar
        
        fig = go.Figure()
        
        # Generate mock 3D coordinates (helix-like structure)
        n = len(sequence)
        t = np.linspace(0, 4*np.pi, n)
        x = np.cos(t)
        y = np.sin(t)
        z = t / (4*np.pi)
        
        # Color by amino acid type
        colors = []
        for aa in sequence:
            if aa in 'AILMFPWYV':  # Hydrophobic
                colors.append('blue')
            elif aa in 'NQSTY':  # Polar
                colors.append('green')
            elif aa in 'DEKR':  # Charged
                colors.append('red')
            else:
                colors.append('gray')
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                showscale=False
            ),
            line=dict(color='lightblue', width=4),
            text=[f"{aa} ({i})" for i, aa in enumerate(sequence)],
            hovertemplate="%{text}<br>Position: %{pointNumber}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Predicted 3D Structure ({len(sequence)} residues)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            height=600
        )
        
        return fig
    
    def plot_contact_map(self, contact_map: np.ndarray, sequence: str = None) -> go.Figure:
        """Create interactive contact map visualization"""
        if contact_map.size == 0:
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=contact_map,
            colorscale='Viridis',
            zmid=0.5,
            colorbar=dict(title="Contact Probability"),
            hovertemplate="Residue %{x} - %{y}<br>Contact Probability: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Predicted Contact Map",
            xaxis_title="Residue Position",
            yaxis_title="Residue Position",
            width=600,
            height=600
        )
        
        return fig
    
    def plot_secondary_structure(self, sequence: str, structure: str, confidence_scores: Dict = None) -> go.Figure:
        """Create interactive secondary structure visualization"""
        if not sequence or not structure:
            return go.Figure()
        
        # Create color mapping for secondary structure
        ss_colors = {'H': 'red', 'E': 'blue', 'C': 'gray'}
        colors = [ss_colors.get(ss, 'black') for ss in structure]
        
        fig = go.Figure()
        
        # Add sequence as text
        for i, (aa, ss) in enumerate(zip(sequence, structure)):
            fig.add_trace(go.Scatter(
                x=[i], y=[0],
                mode='markers+text',
                text=[aa],
                textposition="middle center",
                marker=dict(
                    size=20,
                    color=colors[i],
                    line=dict(width=1, color='black')
                ),
                name=f"Position {i+1}",
                hovertemplate=f"Position: {i+1}<br>Amino Acid: {aa}<br>Structure: {ss}<extra></extra>"
            ))
        
        # Add confidence scores if available
        if confidence_scores:
            fig.add_trace(go.Scatter(
                x=list(range(len(sequence))),
                y=[-0.5] * len(sequence),
                mode='markers',
                marker=dict(
                    size=10,
                    color=[confidence_scores.get('helix', 0.5)] * len(sequence),
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Confidence")
                ),
                name="Confidence"
            ))
        
        fig.update_layout(
            title="Secondary Structure Prediction",
            xaxis_title="Residue Position",
            yaxis=dict(range=[-1, 1], showticklabels=False),
            height=300,
            showlegend=False
        )
        
        return fig
    
    def plot_disorder_analysis(self, sequence: str, disorder_scores: List[float], 
                             disordered_regions: List[Tuple[int, int]] = None) -> go.Figure:
        """Create disorder analysis visualization"""
        if not sequence or not disorder_scores:
            return go.Figure()
        
        fig = go.Figure()
        
        # Plot disorder scores
        fig.add_trace(go.Scatter(
            x=list(range(len(sequence))),
            y=disorder_scores,
            mode='lines+markers',
            name='Disorder Score',
            line=dict(color='purple', width=2),
            marker=dict(size=4)
        ))
        
        # Highlight disordered regions
        if disordered_regions:
            for start, end in disordered_regions:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="purple", opacity=0.2,
                    annotation_text=f"Disordered Region {start}-{end}",
                    annotation_position="top left"
                )
        
        # Add threshold line
        fig.add_hline(y=0.6, line_dash="dash", line_color="red",
                     annotation_text="Disorder Threshold")
        
        fig.update_layout(
            title="Intrinsic Disorder Analysis",
            xaxis_title="Residue Position",
            yaxis_title="Disorder Score",
            height=400
        )
        
        return fig
    
    def plot_stability_landscape(self, sequence: str, mutations: List[Tuple[int, str, str]], 
                               stability_changes: List[float]) -> go.Figure:
        """Create stability landscape visualization"""
        if not mutations or not stability_changes:
            return go.Figure()
        
        positions = [mut[0] for mut in mutations]
        mutation_labels = [f"{mut[1]}{mut[0]+1}{mut[2]}" for mut in mutations]
        
        fig = go.Figure()
        
        # Color code by stability change
        colors = ['red' if change < 0 else 'green' for change in stability_changes]
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=stability_changes,
            mode='markers+text',
            text=mutation_labels,
            textposition="top center",
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='black')
            ),
            name="Mutations",
            hovertemplate="Position: %{x}<br>Mutation: %{text}<br>ΔG: %{y:.3f}<extra></extra>"
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Protein Stability Landscape",
            xaxis_title="Residue Position",
            yaxis_title="Stability Change (ΔG)",
            height=500
        )
        
        return fig
