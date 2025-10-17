#!/usr/bin/env python3
"""
FoldAI Deep Learning - Advanced Protein Structure Predictor
State-of-the-art deep learning models for protein structure prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetchers.uniprot_client import UniProtClient
from analysis.deep_learning_predictor import DeepLearningPredictor
from visualization.sequence_plots import SequenceVisualizer
import pandas as pd
import numpy as np
import torch


def main():
    """Deep learning-powered protein structure prediction"""
    print(" FoldAI Deep Learning - Advanced Protein Structure Predictor")
    print("=" * 65)
    print(" State-of-the-art deep learning models")
    print(" Transformer + CNN + LSTM architecture")
    print(" Trained on thousands of protein structures")
    print()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Initialize deep learning predictor
    print(" Initializing deep learning models...")
    predictor = DeepLearningPredictor(device=device)
    visualizer = SequenceVisualizer()
    uniprot = UniProtClient()
    
    # Get protein data
    print(" Fetching protein data...")
    proteins = uniprot.search_proteins("enzyme", limit=5)
    
    if proteins.empty:
        # Use demo sequences
        sequences = [
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
        ]
        names = ["Demo Protein 1", "Demo Protein 2", "Demo Protein 3"]
    else:
        sequences = proteins['sequence'].dropna().tolist()[:3]
        names = proteins['protein_name'].tolist()[:3]
    
    print(f" Analyzing {len(sequences)} proteins with deep learning...")
    print()
    
    # Deep learning predictions
    for i, (seq, name) in enumerate(zip(sequences, names)):
        print(f" Protein {i+1}: {name}")
        print(f"   Length: {len(seq)} amino acids")
        
        # Deep learning secondary structure prediction
        print("    Deep learning structure prediction...")
        structure_result = predictor.predict_secondary_structure(seq)
        
        # Deep learning stability prediction
        print("   ⚖️  Deep learning stability analysis...")
        stability_result = predictor.predict_stability(seq)
        
        # Deep learning contact map prediction
        print("    Deep learning contact prediction...")
        contact_map = predictor.predict_contact_map(seq)
        
        # Deep learning aggregation prediction
        print("   🔥 Deep learning aggregation analysis...")
        aggregation_result = predictor.predict_aggregation(seq)
        
        # Display results
        print(f"    Structure: {structure_result['helix_content']:.1%} helix, "
              f"{structure_result['sheet_content']:.1%} sheet, "
              f"{structure_result['coil_content']:.1%} coil")
        print(f"   ⚖️  Stability: {stability_result['stability_score']:.3f}/1.0")
        print(f"    Aggregation: {aggregation_result['aggregation_score']:.3f}/1.0")
        print(f"    Confidence: {stability_result['confidence']:.1%}")
        print()
    
    # Generate advanced visualizations
    print(" Creating advanced deep learning visualizations...")
    
    if sequences:
        # Deep learning structure visualization
        structure_result = predictor.predict_secondary_structure(sequences[0])
        structure_fig = visualizer.plot_secondary_structure(
            sequences[0],
            structure_result['structure'],
            {'helix': structure_result['helix_content'], 
             'sheet': structure_result['sheet_content'],
             'coil': structure_result['coil_content']}
        )
        structure_fig.write_html("data/deep_learning_structure.html")
        
        # Deep learning contact map
        contact_map = predictor.predict_contact_map(sequences[0])
        contact_fig = visualizer.plot_contact_map(contact_map)
        contact_fig.write_html("data/deep_learning_contacts.html")
        
        # 3D structure with deep learning features
        structure_3d = visualizer.plot_3d_structure_prediction(sequences[0])
        structure_3d.write_html("data/deep_learning_3d.html")
        
        print("    Saved: data/deep_learning_structure.html")
        print("    Saved: data/deep_learning_contacts.html")
        print("    Saved: data/deep_learning_3d.html")
    
    print()
    print(" Deep Learning Results:")
    print("   • Transformer-based sequence embeddings")
    print("   • CNN + LSTM architecture for structure prediction")
    print("   • Deep neural networks for stability analysis")
    print("   • Advanced contact map prediction")
    print("   • State-of-the-art accuracy")
    print()
    print(" This is the future of protein science!")
    print("   Deep learning models trained on thousands of structures!")
    print()
    print(" Share your results: data/deep_learning_*.html")
    print(" Tag us: #FoldAI #DeepLearning #ProteinScience #AI")


if __name__ == "__main__":
    main()
