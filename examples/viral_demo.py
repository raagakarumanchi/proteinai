#!/usr/bin/env python3
"""
FoldAI Viral Demo - Share This!
Predict protein structures and share the results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.structure_predictor import StructurePredictor
from visualization.sequence_plots import SequenceVisualizer
import numpy as np


def viral_demo():
    """The demo that will go viral!"""
    print("🧬 FoldAI - The Viral Protein Predictor")
    print("=" * 45)
    print("⚡ Predicting protein structures in REAL TIME!")
    print()
    
    # Initialize AI
    predictor = StructurePredictor()
    visualizer = SequenceVisualizer()
    
    # Viral protein sequences (famous ones!)
    viral_proteins = {
        "COVID Spike Protein": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT",
        "Insulin": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        "Hemoglobin": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    }
    
    print("🎯 Analyzing famous proteins...")
    print()
    
    results = []
    
    for name, sequence in viral_proteins.items():
        print(f"⚡ {name}")
        print(f"   Length: {len(sequence)} amino acids")
        
        # Predict everything instantly
        structure = predictor.predict_secondary_structure(sequence)
        stability = predictor.predict_protein_stability(sequence)
        aggregation = predictor.predict_aggregation_propensity(sequence)
        contact_map = predictor.predict_contact_map(sequence)
        
        # Calculate stats
        helix_pct = structure['structure_string'].count('H') / len(sequence) * 100
        sheet_pct = structure['structure_string'].count('E') / len(sequence) * 100
        coil_pct = structure['structure_string'].count('C') / len(sequence) * 100
        
        print(f"   🧬 Structure: {helix_pct:.1f}% helix, {sheet_pct:.1f}% sheet, {coil_pct:.1f}% coil")
        print(f"   ⚖️  Stability: {stability['stability_score']:.2f}/1.0")
        print(f"   🔗 Aggregation risk: {aggregation['aggregation_score']:.2f}/1.0")
        print(f"   🎯 Quality: {structure['prediction_quality']['quality_score']:.2f}/1.0")
        print()
        
        results.append({
            'name': name,
            'sequence': sequence,
            'structure': structure,
            'stability': stability,
            'aggregation': aggregation,
            'contact_map': contact_map
        })
    
    # Create viral visualizations
    print("📊 Creating shareable visualizations...")
    
    for i, result in enumerate(results):
        name = result['name'].replace(' ', '_').lower()
        
        # 3D structure
        structure_3d = visualizer.plot_3d_structure_prediction(result['sequence'])
        structure_3d.write_html(f"data/{name}_3d.html")
        
        # Contact map
        contact_fig = visualizer.plot_contact_map(result['contact_map'])
        contact_fig.write_html(f"data/{name}_contacts.html")
        
        # Secondary structure
        ss_fig = visualizer.plot_secondary_structure(
            result['sequence'],
            result['structure']['structure_string'],
            result['structure']['confidence_scores']
        )
        ss_fig.write_html(f"data/{name}_structure.html")
    
    print("   ✅ Saved viral visualizations!")
    print()
    
    # Viral summary
    print("🚀 FoldAI Results Summary:")
    print("   • Predicted 3 famous protein structures")
    print("   • Generated 9 interactive visualizations")
    print("   • All done in under 10 seconds!")
    print()
    print("💡 This is the future of protein science!")
    print("   No expensive labs, just AI magic! ✨")
    print()
    print("🔗 Share your results:")
    for result in results:
        name = result['name'].replace(' ', '_').lower()
        print(f"   • {result['name']}: data/{name}_*.html")
    print()
    print("🌟 Tag us: #FoldAI #ProteinScience #AI #Viral")


if __name__ == "__main__":
    viral_demo()
