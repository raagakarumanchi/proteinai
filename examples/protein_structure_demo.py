#!/usr/bin/env python3
"""
FoldAI - Protein Structure Analysis Demo
Demonstrates protein structure prediction, stability analysis, and engineering
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetchers.uniprot_client import UniProtClient
from analysis.structure_predictor import StructurePredictor
from analysis.sequence_analyzer import SequenceAnalyzer
from visualization.sequence_plots import SequenceVisualizer
import pandas as pd
import numpy as np


def main():
    print("🧬 FoldAI - Protein Structure Analysis Demo")
    print("=" * 50)
    
    # Initialize components
    uniprot = UniProtClient()
    structure_predictor = StructurePredictor()
    seq_analyzer = SequenceAnalyzer()
    visualizer = SequenceVisualizer()
    
    # 1. Fetch Protein Data
    print("\n📡 Fetching protein data from UniProt...")
    proteins = uniprot.search_proteins("kinase", limit=15)
    
    if proteins.empty:
        print("No proteins found. Using sample data...")
        # Sample protein sequences for demo
        sequences = [
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
        ]
        protein_names = ["Sample Protein 1", "Sample Protein 2", "Sample Protein 3"]
    else:
        sequences = proteins['sequence'].dropna().tolist()
        protein_names = proteins['protein_name'].tolist()
    
    print(f"Analyzing {len(sequences)} protein sequences...")
    
    # 2. Secondary Structure Prediction
    print("\n🔬 Secondary Structure Prediction...")
    for i, (seq, name) in enumerate(zip(sequences[:3], protein_names[:3])):
        print(f"\nAnalyzing {name}:")
        secondary_structure = structure_predictor.predict_secondary_structure(seq)
        
        print(f"  • Sequence length: {len(seq)} amino acids")
        print(f"  • Helix regions: {len(secondary_structure['helix'])}")
        print(f"  • Beta-sheet regions: {len(secondary_structure['sheet'])}")
        print(f"  • Coil regions: {len(secondary_structure['coil'])}")
        
        # Show structure string
        structure_str = secondary_structure['structure_string']
        print(f"  • Structure: {structure_str[:50]}{'...' if len(structure_str) > 50 else ''}")
    
    # 3. Stability Analysis
    print("\n⚖️ Protein Stability Analysis...")
    for i, (seq, name) in enumerate(zip(sequences[:3], protein_names[:3])):
        print(f"\nStability analysis for {name}:")
        stability = structure_predictor.predict_protein_stability(seq)
        
        print(f"  • Stability score: {stability['stability_score']:.3f}")
        print(f"  • Hydrophobic ratio: {stability['features']['hydrophobic_ratio']:.3f}")
        print(f"  • Charged ratio: {stability['features']['charged_ratio']:.3f}")
        print(f"  • Disulfide potential: {stability['features']['disulfide_potential']:.3f}")
    
    # 4. Aggregation Analysis
    print("\n🔗 Aggregation Propensity Analysis...")
    for i, (seq, name) in enumerate(zip(sequences[:3], protein_names[:3])):
        print(f"\nAggregation analysis for {name}:")
        aggregation = structure_predictor.predict_aggregation_propensity(seq)
        
        print(f"  • Aggregation score: {aggregation['aggregation_score']:.3f}")
        print(f"  • Beta-sheet propensity: {aggregation['beta_sheet_propensity']:.3f}")
        print(f"  • Hydrophobicity: {aggregation['hydrophobicity']:.3f}")
        print(f"  • Aggregation hotspots: {len(aggregation['hotspots'])}")
        
        if aggregation['hotspots']:
            print("  • Hotspot regions:")
            for hotspot in aggregation['hotspots'][:3]:
                print(f"    - Positions {hotspot['start']}-{hotspot['end']}: {hotspot['sequence']}")
    
    # 5. Mutation Analysis
    print("\n🧪 Mutation Effect Analysis...")
    test_sequence = sequences[0] if sequences else "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
    
    # Test different types of mutations
    test_mutations = [
        (5, 'L', 'P'),   # Proline (destabilizing)
        (10, 'A', 'V'),  # Conservative
        (15, 'R', 'D'),  # Charge change
        (20, 'H', 'F'),  # Hydrophobicity change
        (25, 'G', 'A'),  # Size change
    ]
    
    print(f"\nTesting mutations in sequence (length: {len(test_sequence)}):")
    mutation_results = structure_predictor.predict_protein_stability(test_sequence, test_mutations)
    
    for effect in mutation_results['mutation_effects']:
        print(f"  • {effect['mutation']}: {effect['effect_type']} (ΔG = {effect['stability_change']:+.3f})")
        print(f"    Hydrophobicity change: {effect['hydrophobicity_change']:+.1f}")
        print(f"    Size change: {effect['size_change']:+.1f}")
        print(f"    Charge change: {effect['charge_change']:+.1f}")
    
    # 6. Functional Domain Analysis
    print("\n🔍 Functional Domain Identification...")
    for i, (seq, name) in enumerate(zip(sequences[:2], protein_names[:2])):
        print(f"\nDomain analysis for {name}:")
        domains = structure_predictor.identify_functional_domains(seq)
        
        print(f"  • Identified domains: {len(domains['domains'])}")
        for domain in domains['domains']:
            print(f"    - {domain['name']}: positions {domain['start']}-{domain['end']}")
            print(f"      Sequence: {domain['sequence']}")
        
        print(f"  • Identified motifs: {len(domains['motifs'])}")
        motif_counts = {}
        for motif in domains['motifs']:
            motif_counts[motif['name']] = motif_counts.get(motif['name'], 0) + 1
        
        for motif_name, count in motif_counts.items():
            print(f"    - {motif_name}: {count} sites")
    
    # 7. Protein Engineering
    print("\n⚙️ Protein Stability Engineering...")
    original_seq = test_sequence
    original_stability = structure_predictor.predict_protein_stability(original_seq)['stability_score']
    
    print(f"Original stability: {original_stability:.3f}")
    
    # Suggest stabilizing mutations
    stabilizing_mutations = [
        (5, 'L', 'I'),   # More hydrophobic
        (10, 'A', 'L'),  # More hydrophobic
        (15, 'R', 'K'),  # Similar charge, more stable
    ]
    
    # Apply mutations
    engineered_seq = list(original_seq)
    for pos, old_aa, new_aa in stabilizing_mutations:
        if pos < len(engineered_seq):
            engineered_seq[pos] = new_aa
    
    engineered_seq = ''.join(engineered_seq)
    engineered_stability = structure_predictor.predict_protein_stability(engineered_seq)['stability_score']
    
    print(f"Engineered stability: {engineered_stability:.3f}")
    print(f"Stability improvement: {engineered_stability - original_stability:+.3f}")
    
    # 8. Generate Visualizations
    print("\n📊 Generating Interactive Visualizations...")
    
    if sequences:
        # Composition plot
        composition_fig = visualizer.plot_amino_acid_composition(
            sequences[:5], 
            [f"Protein {i+1}" for i in range(min(5, len(sequences)))]
        )
        composition_fig.write_html("data/protein_structure_composition.html")
        print("  • Saved: data/protein_structure_composition.html")
        
        # Conservation plot
        conservation_fig = visualizer.plot_conservation_analysis(sequences[:5])
        conservation_fig.write_html("data/protein_structure_conservation.html")
        print("  • Saved: data/protein_structure_conservation.html")
        
        # 3D structure
        structure_fig = visualizer.plot_3d_structure_prediction(sequences[0])
        structure_fig.write_html("data/protein_structure_3d.html")
        print("  • Saved: data/protein_structure_3d.html")
    
    print("\n✅ Protein Structure Analysis Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("  • Secondary structure prediction using Chou-Fasman algorithm")
    print("  • Protein stability analysis and engineering")
    print("  • Aggregation propensity prediction")
    print("  • Mutation effect analysis")
    print("  • Functional domain identification")
    print("  • Interactive 3D structure visualizations")
    print("  • Real-time protein data integration")


if __name__ == "__main__":
    main()
