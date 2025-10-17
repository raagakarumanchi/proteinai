#!/usr/bin/env python3
"""
FoldAI Web App - Viral Protein Structure Predictor
Simple web interface for everyone to use!
"""

from flask import Flask, render_template, request, jsonify, send_file
import sys
import os
import json
import tempfile
import zipfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.structure_predictor import StructurePredictor
from visualization.sequence_plots import SequenceVisualizer
from data_fetchers.uniprot_client import UniProtClient

app = Flask(__name__)
app.config['SECRET_KEY'] = 'foldai-viral-2024'

# Initialize AI components
predictor = StructurePredictor()
visualizer = SequenceVisualizer()
uniprot = UniProtClient()

@app.route('/')
def index():
    """Main page - viral landing page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict protein structure from sequence"""
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Please enter a protein sequence'}), 400
        
        if len(sequence) < 10:
            return jsonify({'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        # Validate sequence (only standard amino acids)
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_aas for aa in sequence):
            return jsonify({'error': 'Invalid amino acids in sequence'}), 400
        
        # Predict structure
        structure = predictor.predict_secondary_structure(sequence)
        stability = predictor.predict_protein_stability(sequence)
        aggregation = predictor.predict_aggregation_propensity(sequence)
        contact_map = predictor.predict_contact_map(sequence)
        
        # Calculate statistics
        helix_pct = structure['structure_string'].count('H') / len(sequence) * 100
        sheet_pct = structure['structure_string'].count('E') / len(sequence) * 100
        coil_pct = structure['structure_string'].count('C') / len(sequence) * 100
        
        # Create visualizations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 3D structure
            structure_3d = visualizer.plot_3d_structure_prediction(sequence)
            structure_3d.write_html(temp_path / "structure_3d.html")
            
            # Contact map
            contact_fig = visualizer.plot_contact_map(contact_map)
            contact_fig.write_html(temp_path / "contact_map.html")
            
            # Secondary structure
            ss_fig = visualizer.plot_secondary_structure(
                sequence,
                structure['structure_string'],
                structure['confidence_scores']
            )
            ss_fig.write_html(temp_path / "secondary_structure.html")
            
            # Create zip file
            zip_path = temp_path / "foldai_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(temp_path / "structure_3d.html", "structure_3d.html")
                zipf.write(temp_path / "contact_map.html", "contact_map.html")
                zipf.write(temp_path / "secondary_structure.html", "secondary_structure.html")
            
            # Read zip file
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
        
        return jsonify({
            'success': True,
            'results': {
                'length': len(sequence),
                'helix_percent': round(helix_pct, 1),
                'sheet_percent': round(sheet_pct, 1),
                'coil_percent': round(coil_pct, 1),
                'stability_score': round(stability['stability_score'], 3),
                'aggregation_score': round(aggregation['aggregation_score'], 3),
                'quality_score': round(structure['prediction_quality']['quality_score'], 3),
                'structure_string': structure['structure_string'],
                'helix_regions': len(structure['helix']),
                'sheet_regions': len(structure['sheet']),
                'coil_regions': len(structure['coil'])
            },
            'zip_data': zip_data.hex()  # Convert to hex for JSON
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/examples')
def examples():
    """Get example protein sequences"""
    examples = {
        'COVID Spike Protein': 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT',
        'Insulin': 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT',
        'Hemoglobin': 'MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH',
        'Green Fluorescent Protein': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    }
    return jsonify(examples)

@app.route('/search')
def search_proteins():
    """Search for proteins by name"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'error': 'Please enter a search term'}), 400
        
        # Search UniProt
        proteins = uniprot.search_proteins(query, limit=10)
        
        if proteins.empty:
            return jsonify({'results': []})
        
        results = []
        for _, protein in proteins.iterrows():
            results.append({
                'name': protein['protein_name'],
                'organism': protein['organism'],
                'length': protein['length'],
                'sequence': protein['sequence'][:100] + '...' if len(protein['sequence']) > 100 else protein['sequence']
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🧬 FoldAI Web App Starting...")
    print("🌐 Open your browser to: http://localhost:5000")
    print("⚡ Ready to predict protein structures!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
