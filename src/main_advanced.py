#!/usr/bin/env python3
"""
FoldAI Advanced Protein Function Predictor - Complete Implementation
Phase 1-3: ESM-2 + GNN + Multi-Modal Fusion + Function-Specific Attention
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetchers.uniprot_client import UniProtClient
from analysis.advanced_protein_predictor import AdvancedProteinPredictorAI
from analysis.structure_predictor import StructurePredictor
from visualization.sequence_plots import SequenceVisualizer
import pandas as pd
import numpy as np
import torch
import time


def main():
    """Complete advanced protein function prediction demonstration"""
    print(" FoldAI Advanced Protein Function Predictor")
    print("=" * 60)
    print(" Phase 1: ESM-2 Transformer Encoder")
    print(" Phase 2: Graph Neural Networks")
    print(" Phase 3: Multi-Modal Fusion + Function-Specific Attention")
    print(" State-of-the-art deep learning architecture")
    print()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Initialize advanced AI system
    print(" Initializing advanced AI systems...")
    start_time = time.time()
    
    advanced_predictor = AdvancedProteinPredictorAI(device=device)
    structure_predictor = StructurePredictor()
    visualizer = SequenceVisualizer()
    uniprot = UniProtClient()
    
    init_time = time.time() - start_time
    print(f"    Initialized in {init_time:.2f} seconds")
    print()
    
    # Get protein data
    print(" Fetching protein data...")
    proteins = uniprot.search_proteins("enzyme", limit=6)
    
    if proteins.empty:
        # Use demo sequences with known functions
        demo_proteins = {
            "Insulin": ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT", "Hormone"),
            "Hemoglobin": ("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH", "Transporter"),
            "Green Fluorescent Protein": ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", "Structural"),
            "COVID Spike Protein": ("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT", "Receptor"),
            "DNA Polymerase": ("MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL", "Enzyme"),
            "Protein Kinase A": ("MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL", "Kinase")
        }
        
        sequences = [seq for seq, _ in demo_proteins.values()]
        names = list(demo_proteins.keys())
        known_functions = [func for _, func in demo_proteins.values()]
    else:
        sequences = proteins['sequence'].dropna().tolist()[:6]
        names = proteins['protein_name'].tolist()[:6]
        known_functions = ["Unknown"] * len(sequences)
    
    print(f" Analyzing {len(sequences)} proteins with advanced AI...")
    print()
    
    # Phase 1: ESM-2 + Structure Prediction
    print(" Phase 1: ESM-2 Transformer + Structure Analysis")
    print("-" * 50)
    
    structure_results = []
    for i, (seq, name) in enumerate(zip(sequences, names)):
        print(f"   Protein {i+1}: {name}")
        print(f"   Length: {len(seq)} amino acids")
        
        # Structure prediction
        structure = structure_predictor.predict_secondary_structure(seq)
        structure_results.append(structure)
        
        helix_pct = structure['structure_string'].count('H') / len(seq) * 100
        sheet_pct = structure['structure_string'].count('E') / len(seq) * 100
        coil_pct = structure['structure_string'].count('C') / len(seq) * 100
        
        print(f"    Structure: {helix_pct:.1f}% helix, {sheet_pct:.1f}% sheet, {coil_pct:.1f}% coil")
        print(f"    Quality: {structure['prediction_quality']['quality_score']:.2f}/1.0")
    print()
    
    # Phase 2: Graph Neural Network Analysis
    print(" Phase 2: Graph Neural Network Analysis")
    print("-" * 45)
    
    graph_analysis = []
    for i, (seq, name) in enumerate(zip(sequences, names)):
        print(f"   Protein {i+1}: {name}")
        
        # Create graph representation
        graph = advanced_predictor.create_protein_graph(seq)
        print(f"    Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
        
        # Analyze graph properties
        node_degrees = torch.bincount(graph.edge_index[0])
        avg_degree = node_degrees.float().mean().item()
        print(f"    Average node degree: {avg_degree:.1f}")
        
        graph_analysis.append(graph)
    print()
    
    # Phase 3: Multi-Modal Fusion + Function Prediction
    print(" Phase 3: Multi-Modal Fusion + Function Prediction")
    print("-" * 55)
    
    prediction_results = []
    for i, (seq, name, structure) in enumerate(zip(sequences, names, structure_results)):
        print(f"   Protein {i+1}: {name}")
        
        # Comprehensive prediction
        start_pred_time = time.time()
        prediction = advanced_predictor.predict_comprehensive(seq, structure['structure_string'])
        pred_time = time.time() - start_pred_time
        
        print(f"    Predicted Function: {prediction['predicted_function']}")
        print(f"    Confidence: {prediction['function_confidence']:.1%}")
        print(f"    Broad Category: {prediction['broad_category']} ({prediction['broad_confidence']:.1%})")
        print(f"   ⚖️  Stability Score: {prediction['stability_score']:.3f}")
        print(f"   🔥 Active Site Probability: {prediction['active_site_probability']:.3f}")
        print(f"    Prediction Time: {pred_time:.2f} seconds")
        
        if known_functions[i] != "Unknown":
            print(f"    Known Function: {known_functions[i]}")
            if prediction['predicted_function'].lower() in known_functions[i].lower():
                print(f"    CORRECT PREDICTION!")
        
        prediction_results.append(prediction)
        print()
    
    # Functional Region Analysis
    print(" Advanced Functional Region Analysis")
    print("-" * 40)
    
    for i, (seq, name, prediction) in enumerate(zip(sequences, names, prediction_results)):
        print(f"   Protein {i+1}: {name}")
        
        # Analyze functional regions
        functional_analysis = advanced_predictor.analyze_functional_regions(seq)
        
        print(f"    Functional Regions: {functional_analysis['total_regions']}")
        print(f"   ⭐ High Importance: {functional_analysis['high_importance_regions']}")
        
        if functional_analysis['functional_regions']:
            top_region = max(functional_analysis['functional_regions'], key=lambda x: x['attention_score'])
            print(f"   🔥 Top Region: {top_region['sequence']} (score: {top_region['attention_score']:.3f})")
        print()
    
    # Generate Advanced Visualizations
    print(" Creating Advanced Multi-Modal Visualizations")
    print("-" * 50)
    
    if sequences:
        # Multi-modal function analysis
        function_fig = visualizer.plot_amino_acid_composition(
            sequences[:4], 
            [f"{name} ({pred['predicted_function']})" for name, pred in zip(names[:4], prediction_results[:4])]
        )
        function_fig.write_html("data/advanced_functions.html")
        
        # Structure + function analysis
        structure_fig = visualizer.plot_secondary_structure(
            sequences[0],
            structure_results[0]['structure_string'],
            structure_results[0]['confidence_scores']
        )
        structure_fig.write_html("data/advanced_structure.html")
        
        # 3D structure with function annotation
        structure_3d = visualizer.plot_3d_structure_prediction(sequences[0])
        structure_3d.write_html("data/advanced_3d.html")
        
        print("    Saved: data/advanced_functions.html")
        print("    Saved: data/advanced_structure.html")
        print("    Saved: data/advanced_3d.html")
    
    # Performance Summary
    total_time = time.time() - start_time
    print(f"\n Advanced AI System Performance Summary")
    print("=" * 50)
    print(f"    Total Runtime: {total_time:.2f} seconds")
    print(f"    Proteins Analyzed: {len(sequences)}")
    print(f"    Average Prediction Time: {total_time/len(sequences):.2f} seconds/protein")
    print(f"    Functions Predicted: {len(set([p['predicted_function'] for p in prediction_results]))}")
    print(f"    Functional Regions Identified: {sum([advanced_predictor.analyze_functional_regions(seq)['total_regions'] for seq in sequences])}")
    print()
    
    print(" Advanced AI Capabilities Demonstrated:")
    print("   • ESM-2 Transformer Encoder (Phase 1)")
    print("   • Graph Neural Networks (Phase 2)")
    print("   • Multi-Modal Fusion (Phase 3)")
    print("   • Function-Specific Attention")
    print("   • Hierarchical Function Prediction")
    print("   • Active Site Detection")
    print("   • Stability Analysis")
    print("   • Functional Region Identification")
    print()
    print(" This is the future of protein science!")
    print("   State-of-the-art AI that understands proteins!")
    print()
    print(" Share your results: data/advanced_*.html")
    print(" Tag us: #FoldAI #AdvancedAI #ProteinScience #DeepLearning")


if __name__ == "__main__":
    main()
