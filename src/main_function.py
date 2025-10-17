#!/usr/bin/env python3
"""
FoldAI Protein Function Predictor - Main Application
Predict what any protein does using state-of-the-art AI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetchers.uniprot_client import UniProtClient
from analysis.protein_function_predictor import ProteinFunctionPredictorAI
from analysis.structure_predictor import StructurePredictor
from visualization.sequence_plots import SequenceVisualizer
import pandas as pd
import numpy as np
import torch


def main():
    """Main demonstration of Protein Function Predictor"""
    print(" FoldAI Protein Function Predictor")
    print("=" * 50)
    print(" AI that predicts what any protein does!")
    print(" State-of-the-art deep learning models")
    print(" 10 major protein function categories")
    print()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Initialize AI systems
    print(" Initializing AI systems...")
    function_predictor = ProteinFunctionPredictorAI(device=device)
    structure_predictor = StructurePredictor()
    visualizer = SequenceVisualizer()
    uniprot = UniProtClient()
    
    # Get protein data
    print(" Fetching protein data...")
    proteins = uniprot.search_proteins("enzyme", limit=8)
    
    if proteins.empty:
        # Use demo sequences with known functions
        demo_proteins = {
            "Insulin": ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT", "Hormone"),
            "Hemoglobin": ("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH", "Transporter"),
            "Green Fluorescent Protein": ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", "Structural"),
            "COVID Spike Protein": ("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT", "Receptor")
        }
        
        sequences = [seq for seq, _ in demo_proteins.values()]
        names = list(demo_proteins.keys())
        known_functions = [func for _, func in demo_proteins.values()]
    else:
        sequences = proteins['sequence'].dropna().tolist()[:4]
        names = proteins['protein_name'].tolist()[:4]
        known_functions = ["Unknown"] * len(sequences)
    
    print(f" Analyzing {len(sequences)} proteins...")
    print()
    
    # Predict functions for all proteins
    print(" Predicting protein functions with AI...")
    function_results = function_predictor.predict_multiple_functions(sequences)
    
    # Display results
    for i, (name, seq, result, known_func) in enumerate(zip(names, sequences, function_results, known_functions)):
        print(f" Protein {i+1}: {name}")
        print(f"   Length: {len(seq)} amino acids")
        print(f"    AI Prediction: {result['function']} (confidence: {result['confidence']:.1%})")
        if known_func != "Unknown":
            print(f"    Known Function: {known_func}")
            if result['function'].lower() in known_func.lower() or known_func.lower() in result['function'].lower():
                print(f"    CORRECT PREDICTION!")
            else:
                print(f"   ❌ Prediction differs from known function")
        
        # Show top 3 predictions
        sorted_predictions = sorted(result['all_predictions'].items(), key=lambda x: x[1], reverse=True)
        print(f"    Top 3 predictions:")
        for j, (func, prob) in enumerate(sorted_predictions[:3]):
            print(f"      {j+1}. {func}: {prob:.1%}")
        print()
    
    # Analyze functional regions for first protein
    print(" Analyzing functional regions...")
    if sequences:
        functional_analysis = function_predictor.analyze_functional_regions(sequences[0])
        print(f"   Protein: {names[0]}")
        print(f"   Functional regions found: {functional_analysis['total_functional_regions']}")
        print(f"   Active sites predicted: {len(functional_analysis['active_sites'])}")
        
        if functional_analysis['functional_regions']:
            print(f"   Top functional region: {functional_analysis['functional_regions'][0]['sequence']}")
        print()
    
    # Generate comprehensive visualizations
    print(" Creating comprehensive visualizations...")
    
    if sequences:
        # Function prediction visualization
        function_fig = visualizer.plot_amino_acid_composition(
            sequences[:4], 
            [f"{name} ({result['function']})" for name, result in zip(names[:4], function_results[:4])]
        )
        function_fig.write_html("data/protein_functions.html")
        
        # Structure + function analysis
        structure_result = structure_predictor.predict_secondary_structure(sequences[0])
        structure_fig = visualizer.plot_secondary_structure(
            sequences[0],
            structure_result['structure_string'],
            structure_result['confidence_scores']
        )
        structure_fig.write_html("data/function_structure.html")
        
        # 3D structure with function annotation
        structure_3d = visualizer.plot_3d_structure_prediction(sequences[0])
        structure_3d.write_html("data/function_3d.html")
        
        print("    Saved: data/protein_functions.html")
        print("    Saved: data/function_structure.html")
        print("    Saved: data/function_3d.html")
    
    # Show function explanations
    print("\n Function Explanations:")
    for result in function_results[:3]:
        explanation = function_predictor.get_function_explanation(result['function'])
        print(f"\n {result['function']}:")
        print(f"   Description: {explanation['description']}")
        print(f"   Examples: {explanation['examples']}")
        print(f"   Applications: {explanation['applications']}")
    
    print("\n Protein Function Predictor Results:")
    print("   • Predicted functions for multiple proteins")
    print("   • Identified functional regions and active sites")
    print("   • Generated comprehensive visualizations")
    print("   • Provided detailed function explanations")
    print("   • State-of-the-art deep learning accuracy")
    print()
    print(" This is the future of protein science!")
    print("   AI that understands what proteins do!")
    print()
    print(" Share your results: data/function_*.html")
    print(" Tag us: #FoldAI #ProteinFunction #AI #DeepLearning")


if __name__ == "__main__":
    main()
