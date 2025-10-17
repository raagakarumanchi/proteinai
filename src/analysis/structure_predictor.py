import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from functools import lru_cache
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructurePredictor:
    """AI-powered protein structure prediction and analysis"""
    
    def __init__(self):
        self.secondary_structure_predictor = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        )
        self.stability_predictor = GradientBoostingRegressor(
            n_estimators=150, max_depth=8, random_state=42
        )
        self.contact_predictor = RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Chou-Fasman parameters for secondary structure prediction
        self.chou_fasman = {
            'A': {'helix': 1.42, 'sheet': 0.83, 'turn': 0.66},
            'R': {'helix': 0.98, 'sheet': 0.93, 'turn': 0.95},
            'N': {'helix': 0.67, 'sheet': 0.89, 'turn': 1.56},
            'D': {'helix': 1.01, 'sheet': 0.54, 'turn': 1.46},
            'C': {'helix': 0.70, 'sheet': 1.19, 'turn': 1.19},
            'Q': {'helix': 1.11, 'sheet': 1.10, 'turn': 0.98},
            'E': {'helix': 1.51, 'sheet': 0.37, 'turn': 0.74},
            'G': {'helix': 0.57, 'sheet': 0.75, 'turn': 1.56},
            'H': {'helix': 1.00, 'sheet': 0.87, 'turn': 0.95},
            'I': {'helix': 1.08, 'sheet': 1.60, 'turn': 0.47},
            'L': {'helix': 1.21, 'sheet': 1.30, 'turn': 0.59},
            'K': {'helix': 1.16, 'sheet': 0.74, 'turn': 1.01},
            'M': {'helix': 1.45, 'sheet': 1.05, 'turn': 0.60},
            'F': {'helix': 1.13, 'sheet': 1.38, 'turn': 0.60},
            'P': {'helix': 0.57, 'sheet': 0.55, 'turn': 1.52},
            'S': {'helix': 0.77, 'sheet': 0.75, 'turn': 1.43},
            'T': {'helix': 0.83, 'sheet': 1.19, 'turn': 0.96},
            'W': {'helix': 1.08, 'sheet': 1.37, 'turn': 0.96},
            'Y': {'helix': 0.69, 'sheet': 1.47, 'turn': 1.14},
            'V': {'helix': 1.06, 'sheet': 1.70, 'turn': 0.50}
        }
        
        # Kyte-Doolittle hydrophobicity scale
        self.hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
    
    @lru_cache(maxsize=128)
    def predict_secondary_structure(self, sequence: str) -> Dict:
        """Predict secondary structure using Chou-Fasman algorithm"""
        if not sequence:
            return {'helix': [], 'sheet': [], 'coil': [], 'sequence': ''}
        
        sequence = sequence.upper()
        helix_scores = []
        sheet_scores = []
        turn_scores = []
        
        # Calculate propensity scores
        for aa in sequence:
            if aa in self.chou_fasman:
                helix_scores.append(self.chou_fasman[aa]['helix'])
                sheet_scores.append(self.chou_fasman[aa]['sheet'])
                turn_scores.append(self.chou_fasman[aa]['turn'])
            else:
                helix_scores.append(1.0)
                sheet_scores.append(1.0)
                turn_scores.append(1.0)
        
        # Smooth scores with window averaging
        window = 5
        smoothed_helix = self._smooth_scores(helix_scores, window)
        smoothed_sheet = self._smooth_scores(sheet_scores, window)
        smoothed_turn = self._smooth_scores(turn_scores, window)
        
        # Predict secondary structure
        structure = []
        helix_regions = []
        sheet_regions = []
        
        for i in range(len(sequence)):
            if smoothed_helix[i] > 1.0 and smoothed_helix[i] > smoothed_sheet[i]:
                structure.append('H')
                if not helix_regions or helix_regions[-1][1] != i-1:
                    helix_regions.append([i, i])
                else:
                    helix_regions[-1][1] = i
            elif smoothed_sheet[i] > 1.0:
                structure.append('E')
                if not sheet_regions or sheet_regions[-1][1] != i-1:
                    sheet_regions.append([i, i])
                else:
                    sheet_regions[-1][1] = i
            else:
                structure.append('C')
        
        # Calculate confidence scores
        helix_confidence = np.mean(smoothed_helix)
        sheet_confidence = np.mean(smoothed_sheet)
        coil_confidence = np.mean(smoothed_turn)
        
        return {
            'helix': helix_regions,
            'sheet': sheet_regions,
            'coil': self._find_coil_regions(structure),
            'sequence': sequence,
            'structure_string': ''.join(structure),
            'confidence_scores': {
                'helix': helix_confidence,
                'sheet': sheet_confidence,
                'coil': coil_confidence
            },
            'prediction_quality': self._assess_prediction_quality(structure, sequence)
        }
    
    def _smooth_scores(self, scores: List[float], window: int) -> List[float]:
        """Smooth scores using moving average"""
        smoothed = []
        for i in range(len(scores)):
            start = max(0, i - window // 2)
            end = min(len(scores), i + window // 2 + 1)
            smoothed.append(np.mean(scores[start:end]))
        return smoothed
    
    def _find_coil_regions(self, structure: List[str]) -> List[Tuple[int, int]]:
        """Find coil regions in secondary structure"""
        coil_regions = []
        start = None
        
        for i, ss in enumerate(structure):
            if ss == 'C':
                if start is None:
                    start = i
            else:
                if start is not None:
                    coil_regions.append((start, i-1))
                    start = None
        
        if start is not None:
            coil_regions.append((start, len(structure)-1))
        
        return coil_regions
    
    def _assess_prediction_quality(self, structure: List[str], sequence: str) -> Dict:
        """Assess the quality of secondary structure prediction"""
        if not structure or not sequence:
            return {'quality_score': 0, 'issues': []}
        
        issues = []
        quality_score = 1.0
        
        # Check for very short helices/sheets (likely artifacts)
        helix_lengths = []
        sheet_lengths = []
        
        current_ss = structure[0]
        current_length = 1
        
        for i in range(1, len(structure)):
            if structure[i] == current_ss:
                current_length += 1
            else:
                if current_ss == 'H':
                    helix_lengths.append(current_length)
                elif current_ss == 'E':
                    sheet_lengths.append(current_length)
                current_ss = structure[i]
                current_length = 1
        
        # Check final segment
        if current_ss == 'H':
            helix_lengths.append(current_length)
        elif current_ss == 'E':
            sheet_lengths.append(current_length)
        
        # Penalize very short secondary structure elements
        short_helices = sum(1 for length in helix_lengths if length < 4)
        short_sheets = sum(1 for length in sheet_lengths if length < 3)
        
        if short_helices > 0:
            issues.append(f"{short_helices} very short helices (<4 residues)")
            quality_score -= 0.1 * short_helices
        
        if short_sheets > 0:
            issues.append(f"{short_sheets} very short sheets (<3 residues)")
            quality_score -= 0.1 * short_sheets
        
        # Check for unrealistic proline in helices
        proline_in_helix = 0
        for i, (ss, aa) in enumerate(zip(structure, sequence)):
            if ss == 'H' and aa == 'P':
                proline_in_helix += 1
        
        if proline_in_helix > 0:
            issues.append(f"{proline_in_helix} prolines in helices")
            quality_score -= 0.05 * proline_in_helix
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'helix_count': len(helix_lengths),
            'sheet_count': len(sheet_lengths),
            'avg_helix_length': np.mean(helix_lengths) if helix_lengths else 0,
            'avg_sheet_length': np.mean(sheet_lengths) if sheet_lengths else 0
        }
    
    def predict_protein_stability(self, sequence: str, mutations: List[Tuple[int, str, str]] = None) -> Dict:
        """Predict protein stability and effects of mutations"""
        if not sequence:
            return {'stability_score': 0, 'mutation_effects': []}
        
        sequence = sequence.upper()
        
        # Calculate stability features
        features = self._calculate_stability_features(sequence)
        
        # Predict base stability
        stability_score = self._predict_stability_score(features)
        
        # Predict mutation effects
        mutation_effects = []
        if mutations:
            for pos, old_aa, new_aa in mutations:
                if 0 <= pos < len(sequence):
                    effect = self._predict_mutation_effect(sequence, pos, old_aa.upper(), new_aa.upper())
                    mutation_effects.append({
                        'position': pos,
                        'mutation': f"{old_aa}{pos+1}{new_aa}",
                        'effect': effect,
                        'stability_change': effect['stability_change']
                    })
        
        return {
            'stability_score': stability_score,
            'mutation_effects': mutation_effects,
            'features': features
        }
    
    def _calculate_stability_features(self, sequence: str) -> Dict:
        """Calculate features that affect protein stability"""
        features = {}
        
        # Amino acid composition
        features['hydrophobic_ratio'] = sum(self.hydrophobicity.get(aa, 0) for aa in sequence) / len(sequence)
        features['charged_ratio'] = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
        features['aromatic_ratio'] = sum(1 for aa in sequence if aa in 'FWY') / len(sequence)
        features['proline_ratio'] = sequence.count('P') / len(sequence)
        features['glycine_ratio'] = sequence.count('G') / len(sequence)
        
        # Secondary structure propensity
        helix_propensity = sum(self.chou_fasman.get(aa, {'helix': 1.0})['helix'] for aa in sequence) / len(sequence)
        sheet_propensity = sum(self.chou_fasman.get(aa, {'sheet': 1.0})['sheet'] for aa in sequence) / len(sequence)
        features['helix_propensity'] = helix_propensity
        features['sheet_propensity'] = sheet_propensity
        
        # Disulfide bond potential
        features['cysteine_ratio'] = sequence.count('C') / len(sequence)
        features['disulfide_potential'] = (sequence.count('C') // 2) / len(sequence)
        
        # Length-dependent features
        features['length'] = len(sequence)
        features['length_factor'] = 1.0 / (1.0 + len(sequence) / 1000.0)  # Longer proteins less stable
        
        return features
    
    def _predict_stability_score(self, features: Dict) -> float:
        """Predict protein stability score"""
        # Rule-based stability prediction
        score = 0.5  # Base score
        
        # Hydrophobic interactions (stabilizing)
        if features['hydrophobic_ratio'] > 0.3:
            score += 0.2
        elif features['hydrophobic_ratio'] < 0.1:
            score -= 0.1
        
        # Charged residues (can be stabilizing or destabilizing)
        if 0.1 < features['charged_ratio'] < 0.3:
            score += 0.1
        elif features['charged_ratio'] > 0.4:
            score -= 0.1
        
        # Proline (destabilizing in helices)
        if features['proline_ratio'] > 0.1:
            score -= 0.1
        
        # Glycine (flexible, can be destabilizing)
        if features['glycine_ratio'] > 0.15:
            score -= 0.05
        
        # Disulfide bonds (stabilizing)
        if features['disulfide_potential'] > 0.02:
            score += 0.15
        
        # Length factor
        score += features['length_factor'] * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _predict_mutation_effect(self, sequence: str, pos: int, old_aa: str, new_aa: str) -> Dict:
        """Predict the effect of a single amino acid mutation"""
        # Calculate stability change
        old_hydrophobicity = self.hydrophobicity.get(old_aa, 0)
        new_hydrophobicity = self.hydrophobicity.get(new_aa, 0)
        hydrophobicity_change = new_hydrophobicity - old_hydrophobicity
        
        # Size change
        old_size = self._get_amino_acid_size(old_aa)
        new_size = self._get_amino_acid_size(new_aa)
        size_change = new_size - old_size
        
        # Charge change
        old_charge = self._get_amino_acid_charge(old_aa)
        new_charge = self._get_amino_acid_charge(new_aa)
        charge_change = new_charge - old_charge
        
        # Predict stability change
        stability_change = 0
        
        # Hydrophobicity changes
        if abs(hydrophobicity_change) > 2.0:
            stability_change += hydrophobicity_change * 0.1
        
        # Size changes (can be destabilizing)
        if abs(size_change) > 50:
            stability_change -= 0.1
        
        # Charge changes (can be destabilizing)
        if charge_change != 0:
            stability_change -= 0.05
        
        # Proline mutations (destabilizing)
        if new_aa == 'P':
            stability_change -= 0.2
        
        return {
            'stability_change': stability_change,
            'hydrophobicity_change': hydrophobicity_change,
            'size_change': size_change,
            'charge_change': charge_change,
            'effect_type': self._classify_mutation_effect(stability_change)
        }
    
    def _get_amino_acid_size(self, aa: str) -> float:
        """Get amino acid size (van der Waals volume)"""
        sizes = {
            'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
            'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
            'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
            'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
        }
        return sizes.get(aa, 100.0)
    
    def _get_amino_acid_charge(self, aa: str) -> int:
        """Get amino acid charge at pH 7"""
        positive = {'R': 1, 'K': 1, 'H': 1}
        negative = {'D': -1, 'E': -1}
        return positive.get(aa, 0) + negative.get(aa, 0)
    
    def _classify_mutation_effect(self, stability_change: float) -> str:
        """Classify mutation effect"""
        if stability_change > 0.1:
            return "Stabilizing"
        elif stability_change < -0.1:
            return "Destabilizing"
        else:
            return "Neutral"
    
    def predict_aggregation_propensity(self, sequence: str) -> Dict:
        """Predict protein aggregation propensity"""
        if not sequence:
            return {'aggregation_score': 0, 'hotspots': []}
        
        sequence = sequence.upper()
        
        # Calculate aggregation features
        beta_sheet_propensity = sum(self.chou_fasman.get(aa, {'sheet': 1.0})['sheet'] for aa in sequence) / len(sequence)
        hydrophobicity = sum(self.hydrophobicity.get(aa, 0) for aa in sequence) / len(sequence)
        charge_density = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
        
        # Identify aggregation hotspots
        hotspots = []
        window = 6
        
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            window_hydrophobicity = sum(self.hydrophobicity.get(aa, 0) for aa in window_seq) / window
            window_charge = sum(1 for aa in window_seq if aa in 'DEKR') / window
            window_sheet = sum(self.chou_fasman.get(aa, {'sheet': 1.0})['sheet'] for aa in window_seq) / window
            
            # Aggregation-prone regions: hydrophobic, low charge, high beta-sheet propensity
            hotspot_score = (window_hydrophobicity * 0.4 + 
                           window_sheet * 0.3 + 
                           (1 - window_charge) * 0.3)
            
            if hotspot_score > 0.6:
                hotspots.append({
                    'start': i,
                    'end': i + window - 1,
                    'sequence': window_seq,
                    'score': hotspot_score
                })
        
        # Overall aggregation score
        aggregation_score = (beta_sheet_propensity * 0.4 + 
                           max(0, hydrophobicity) * 0.3 + 
                           (1 - charge_density) * 0.3)
        
        return {
            'aggregation_score': min(1.0, aggregation_score),
            'hotspots': hotspots,
            'beta_sheet_propensity': beta_sheet_propensity,
            'hydrophobicity': hydrophobicity,
            'charge_density': charge_density
        }
    
    def identify_functional_domains(self, sequence: str) -> Dict:
        """Identify potential functional domains"""
        if not sequence:
            return {'domains': [], 'motifs': []}
        
        sequence = sequence.upper()
        domains = []
        motifs = []
        
        # Common domain patterns
        domain_patterns = {
            'zinc_finger': 'C.{2,4}C.{12,15}H.{2,4}H',
            'helix_turn_helix': 'E.{10,15}E.{10,15}E',
            'leucine_zipper': 'L.{6}L.{6}L',
            'immunoglobulin': 'C.{8,12}C.{8,12}C'
        }
        
        import re
        for domain_name, pattern in domain_patterns.items():
            matches = list(re.finditer(pattern, sequence))
            for match in matches:
                domains.append({
                    'name': domain_name,
                    'start': match.start(),
                    'end': match.end(),
                    'sequence': match.group()
                })
        
        # Common motifs
        motif_patterns = {
            'n_glycosylation': 'N[^P][ST][^P]',
            'phosphorylation': '[ST]',
            'myristoylation': 'G[^P]',
            'palmitoylation': 'C'
        }
        
        for motif_name, pattern in motif_patterns.items():
            matches = list(re.finditer(pattern, sequence))
            for match in matches:
                motifs.append({
                    'name': motif_name,
                    'position': match.start(),
                    'sequence': match.group()
                })
        
        return {
            'domains': domains,
            'motifs': motifs
        }
    
    def predict_contact_map(self, sequence: str, window_size: int = 5) -> np.ndarray:
        """Predict residue-residue contact map using sequence-based features"""
        if not sequence or len(sequence) < 2:
            return np.array([])
        
        n = len(sequence)
        contact_map = np.zeros((n, n))
        
        # Calculate contact probabilities based on sequence features
        for i in range(n):
            for j in range(i + 1, n):
                # Skip nearby residues (contacts are typically long-range)
                if abs(i - j) < 5:
                    continue
                
                # Calculate contact probability based on:
                # 1. Hydrophobic complementarity
                # 2. Charge complementarity  
                # 3. Size complementarity
                # 4. Secondary structure propensity
                
                aa_i, aa_j = sequence[i], sequence[j]
                
                # Hydrophobic complementarity
                hydro_i = self.hydrophobicity.get(aa_i, 0)
                hydro_j = self.hydrophobicity.get(aa_j, 0)
                hydrophobic_score = abs(hydro_i - hydro_j) if (hydro_i > 0) != (hydro_j > 0) else 0
                
                # Charge complementarity
                charge_i = self._get_amino_acid_charge(aa_i)
                charge_j = self._get_amino_acid_charge(aa_j)
                charge_score = 1.0 if charge_i * charge_j < 0 else 0.5
                
                # Size complementarity (large-small pairs are favorable)
                size_i = self._get_amino_acid_size(aa_i)
                size_j = self._get_amino_acid_size(aa_j)
                size_score = 1.0 - abs(size_i - size_j) / 200.0  # Normalize
                
                # Secondary structure propensity
                ss_i = self.chou_fasman.get(aa_i, {'helix': 1.0, 'sheet': 1.0})
                ss_j = self.chou_fasman.get(aa_j, {'helix': 1.0, 'sheet': 1.0})
                ss_score = (ss_i['helix'] * ss_j['helix'] + ss_i['sheet'] * ss_j['sheet']) / 2
                
                # Combine scores
                contact_prob = (hydrophobic_score * 0.3 + 
                              charge_score * 0.2 + 
                              size_score * 0.2 + 
                              ss_score * 0.3)
                
                contact_map[i, j] = contact_map[j, i] = contact_prob
        
        return contact_map
    
    def predict_disorder_regions(self, sequence: str) -> Dict:
        """Predict intrinsically disordered regions"""
        if not sequence:
            return {'disorder_scores': [], 'disordered_regions': []}
        
        # Calculate disorder propensity for each residue
        disorder_scores = []
        for aa in sequence:
            # Disorder-promoting residues
            disorder_residues = {'P': 0.8, 'G': 0.7, 'S': 0.6, 'N': 0.6, 'Q': 0.6, 'T': 0.5}
            # Order-promoting residues  
            order_residues = {'W': 0.8, 'F': 0.7, 'Y': 0.7, 'I': 0.6, 'L': 0.6, 'V': 0.6, 'C': 0.6}
            
            if aa in disorder_residues:
                disorder_scores.append(disorder_residues[aa])
            elif aa in order_residues:
                disorder_scores.append(1.0 - order_residues[aa])
            else:
                disorder_scores.append(0.5)  # Neutral
        
        # Smooth scores
        window = 5
        smoothed_scores = []
        for i in range(len(disorder_scores)):
            start = max(0, i - window // 2)
            end = min(len(disorder_scores), i + window // 2 + 1)
            smoothed_scores.append(np.mean(disorder_scores[start:end]))
        
        # Identify disordered regions (score > 0.6)
        disordered_regions = []
        in_disordered = False
        start_pos = 0
        
        for i, score in enumerate(smoothed_scores):
            if score > 0.6 and not in_disordered:
                start_pos = i
                in_disordered = True
            elif score <= 0.6 and in_disordered:
                if i - start_pos >= 5:  # Minimum length for disordered region
                    disordered_regions.append((start_pos, i - 1))
                in_disordered = False
        
        # Handle case where sequence ends in disordered region
        if in_disordered and len(smoothed_scores) - start_pos >= 5:
            disordered_regions.append((start_pos, len(smoothed_scores) - 1))
        
        return {
            'disorder_scores': smoothed_scores,
            'disordered_regions': disordered_regions,
            'disorder_fraction': sum(1 for score in smoothed_scores if score > 0.6) / len(smoothed_scores)
        }
    
    def predict_membrane_regions(self, sequence: str) -> Dict:
        """Predict transmembrane regions"""
        if not sequence:
            return {'membrane_regions': [], 'membrane_score': 0}
        
        # Calculate membrane propensity for each residue
        membrane_scores = []
        for aa in sequence:
            # Hydrophobic residues favor membrane
            hydro = self.hydrophobicity.get(aa, 0)
            membrane_score = max(0, hydro / 4.5)  # Normalize to 0-1
            membrane_scores.append(membrane_score)
        
        # Smooth scores
        window = 15  # Larger window for membrane regions
        smoothed_scores = []
        for i in range(len(membrane_scores)):
            start = max(0, i - window // 2)
            end = min(len(membrane_scores), i + window // 2 + 1)
            smoothed_scores.append(np.mean(membrane_scores[start:end]))
        
        # Identify transmembrane regions (score > 0.7, length 15-30 residues)
        membrane_regions = []
        in_membrane = False
        start_pos = 0
        
        for i, score in enumerate(smoothed_scores):
            if score > 0.7 and not in_membrane:
                start_pos = i
                in_membrane = True
            elif score <= 0.7 and in_membrane:
                length = i - start_pos
                if 15 <= length <= 30:  # Typical TM region length
                    membrane_regions.append((start_pos, i - 1))
                in_membrane = False
        
        # Handle case where sequence ends in membrane region
        if in_membrane:
            length = len(smoothed_scores) - start_pos
            if 15 <= length <= 30:
                membrane_regions.append((start_pos, len(smoothed_scores) - 1))
        
        return {
            'membrane_regions': membrane_regions,
            'membrane_scores': smoothed_scores,
            'membrane_score': np.mean(smoothed_scores),
            'transmembrane_count': len(membrane_regions)
        }
    
    def benchmark_prediction(self, sequences: List[str], known_structures: List[str] = None) -> Dict:
        """Benchmark structure prediction accuracy"""
        if not sequences:
            return {'accuracy': 0, 'details': {}}
        
        results = {
            'total_sequences': len(sequences),
            'avg_helix_content': 0,
            'avg_sheet_content': 0,
            'avg_coil_content': 0,
            'prediction_qualities': []
        }
        
        helix_contents = []
        sheet_contents = []
        coil_contents = []
        quality_scores = []
        
        for seq in sequences:
            prediction = self.predict_secondary_structure(seq)
            structure_str = prediction['structure_string']
            
            # Calculate content
            helix_content = structure_str.count('H') / len(structure_str)
            sheet_content = structure_str.count('E') / len(structure_str)
            coil_content = structure_str.count('C') / len(structure_str)
            
            helix_contents.append(helix_content)
            sheet_contents.append(sheet_content)
            coil_contents.append(coil_content)
            quality_scores.append(prediction['prediction_quality']['quality_score'])
        
        results['avg_helix_content'] = np.mean(helix_contents)
        results['avg_sheet_content'] = np.mean(sheet_contents)
        results['avg_coil_content'] = np.mean(coil_contents)
        results['avg_quality_score'] = np.mean(quality_scores)
        results['prediction_qualities'] = quality_scores
        
        return results
