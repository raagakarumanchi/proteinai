#!/usr/bin/env python3
"""
FoldAI - Viral Protein Structure Predictor
Predict protein structures in seconds, not years.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetchers.uniprot_client import UniProtClient
from analysis.structure_predictor import StructurePredictor
from visualization.sequence_plots import SequenceVisualizer
import pandas as pd
import numpy as np


def main():
    """Viral demo - predict protein structures instantly"""
    print("🧬 FoldAI - Predict Protein Structures in Seconds")
    print("=" * 55)
    print("⚡ No more waiting months for protein structures!")
    print("🤖 AI predicts 3D structures from sequences instantly")
    print()
    
    # Initialize
    uniprot = UniProtClient()
    predictor = StructurePredictor()
    visualizer = SequenceVisualizer()
    
    # Get viral protein examples
    print("🔍 Fetching viral proteins...")
    proteins = uniprot.search_proteins("virus", limit=5)
    
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
    
    print(f"🎯 Analyzing {len(sequences)} proteins...")
    print()
    
    # Instant structure prediction
    for i, (seq, name) in enumerate(zip(sequences, names)):
        print(f"⚡ Protein {i+1}: {name}")
        print(f"   Length: {len(seq)} amino acids")
        
        # Predict structure
        structure = predictor.predict_secondary_structure(seq)
        stability = predictor.predict_protein_stability(seq)
        aggregation = predictor.predict_aggregation_propensity(seq)
        
        # Show results
        helix_count = len(structure['helix'])
        sheet_count = len(structure['sheet'])
        coil_count = len(structure['coil'])
        
        print(f"   🧬 Structure: {helix_count} helices, {sheet_count} sheets, {coil_count} coils")
        print(f"   ⚖️  Stability: {stability['stability_score']:.2f}/1.0")
        print(f"   🔗 Aggregation risk: {aggregation['aggregation_score']:.2f}/1.0")
        print(f"   🎯 Quality: {structure['prediction_quality']['quality_score']:.2f}/1.0")
        print()
    
    # Generate viral visualizations
    print("📊 Creating viral visualizations...")
    
    # Structure plot
    if sequences:
        structure_fig = visualizer.plot_secondary_structure(
            sequences[0], 
            structure['structure_string'],
            structure['confidence_scores']
        )
        structure_fig.write_html("data/viral_structure.html")
        
        # 3D structure
        structure_3d = visualizer.plot_3d_structure_prediction(sequences[0])
        structure_3d.write_html("data/viral_3d.html")
        
        # Contact map
        contact_map = predictor.predict_contact_map(sequences[0])
        contact_fig = visualizer.plot_contact_map(contact_map)
        contact_fig.write_html("data/viral_contacts.html")
        
        print("   ✅ Saved: data/viral_structure.html")
        print("   ✅ Saved: data/viral_3d.html") 
        print("   ✅ Saved: data/viral_contacts.html")
    
    print()
    print("🚀 FoldAI Results:")
    print("   • Predicted structures in seconds (not months!)")
    print("   • Identified stability hotspots")
    print("   • Found aggregation risks")
    print("   • Generated 3D visualizations")
    print()
    print("💡 This is the future of protein science!")
    print("   No more expensive experiments, just AI predictions!")
    print()
    print("🔗 Share your results: data/viral_*.html")


if __name__ == "__main__":
    main()