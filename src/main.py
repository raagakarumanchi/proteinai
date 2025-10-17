#!/usr/bin/env python3
"""
FoldAI - Protein Intelligence CLI
Analyze protein sequences for structure and function from the command line.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetchers.uniprot_client import UniProtClient
from analysis.structure_predictor import StructurePredictor
from visualization.sequence_plots import SequenceVisualizer


DEFAULT_DEMO: List[Tuple[str, str]] = [
    ("Demo Protein 1", "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"),
    ("Demo Protein 2", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    ("Demo Protein 3", "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FoldAI - Predict protein structure and function from sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["structure", "function", "both"],
        default="structure",
        help="Select prediction mode",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="UniProt search query to fetch proteins",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=5,
        help="Number of UniProt results to use",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        action="append",
        help="Provide an amino acid sequence directly (can be repeated)",
    )
    parser.add_argument(
        "-f",
        "--fasta",
        action="append",
        help="Path to FASTA file containing sequences (can be repeated)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data",
        help="Directory where visualizations will be saved",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip generation of HTML visualizations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top function predictions to display",
    )
    return parser.parse_args()


def sanitize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def slugify(name: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name)
    return slug or "protein"


def load_fasta_file(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_name: str = ""
    current_sequence: List[str] = []
    file_path = Path(path)

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_sequence:
                        records.append((current_name or file_path.stem, sanitize_sequence("".join(current_sequence))))
                        current_sequence = []
                    current_name = line[1:].strip()
                else:
                    current_sequence.append(line)
            if current_sequence:
                records.append((current_name or file_path.stem, sanitize_sequence("".join(current_sequence))))
    except OSError as exc:
        print(f"Could not read FASTA file '{path}': {exc}")

    return [(name, seq) for name, seq in records if seq]


def ensure_unique_names(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: Dict[str, int] = {}
    unique: List[Tuple[str, str]] = []

    for name, sequence in items:
        base = name or "Protein"
        count = seen.get(base, 0)
        unique_name = base if count == 0 else f"{base} ({count + 1})"
        seen[base] = count + 1
        unique.append((unique_name, sequence))

    return unique


def gather_sequences(args: argparse.Namespace, uniprot: UniProtClient) -> List[Tuple[str, str]]:
    collected: List[Tuple[str, str]] = []

    if args.sequence:
        for idx, seq in enumerate(args.sequence, start=1):
            clean = sanitize_sequence(seq)
            if clean:
                collected.append((f"Input Sequence {idx}", clean))

    if args.fasta:
        for fasta_path in args.fasta:
            collected.extend(load_fasta_file(fasta_path))

    if args.query:
        proteins = uniprot.search_proteins(args.query, limit=args.limit)
        if not proteins.empty:
            for entry in proteins.itertuples():
                sequence = sanitize_sequence(getattr(entry, "sequence", "") or "")
                if not sequence:
                    continue
                name = getattr(entry, "protein_name", "") or getattr(entry, "id", "") or getattr(entry, "accession", "")
                collected.append((name, sequence))

    collected = [(name, seq) for name, seq in collected if seq]
    return ensure_unique_names(collected)


def print_structure_summary(structure: Dict, stability: Dict, aggregation: Dict) -> None:
    helix_count = len(structure.get("helix", []))
    sheet_count = len(structure.get("sheet", []))
    coil_count = len(structure.get("coil", []))
    quality = structure.get("prediction_quality", {}).get("quality_score", 0.0)
    stability_score = stability.get("stability_score", 0.0)
    aggregation_score = aggregation.get("aggregation_score", 0.0)

    print(f"  Structure: {helix_count} helices, {sheet_count} sheets, {coil_count} coils")
    print(f"  Stability: {stability_score:.2f}/1.0")
    print(f"  Aggregation risk: {aggregation_score:.2f}/1.0")
    print(f"  Quality: {quality:.2f}/1.0")


def print_function_summary(prediction: Dict, top_k: int) -> None:
    function = prediction.get("function", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    print(f"  Function: {function} ({confidence:.1%})")

    all_predictions = prediction.get("all_predictions", {})
    if all_predictions:
        print("  Top predictions:")
        sorted_preds = sorted(all_predictions.items(), key=lambda item: item[1], reverse=True)
        for label, prob in sorted_preds[:top_k]:
            print(f"    - {label}: {prob:.1%}")


def generate_structure_visuals(visualizer: SequenceVisualizer, output_dir: Path, summary: Dict, contact_map) -> None:
    sequence = summary["sequence"]
    structure = summary["structure"]
    name = summary["name"]
    slug = slugify(name)

    structure_fig = visualizer.plot_secondary_structure(
        sequence,
        structure["structure_string"],
        structure["confidence_scores"],
    )
    structure_path = output_dir / f"{slug}_structure.html"
    structure_fig.write_html(structure_path.as_posix())

    structure_3d = visualizer.plot_3d_structure_prediction(sequence)
    structure_3d_path = output_dir / f"{slug}_structure_3d.html"
    structure_3d.write_html(structure_3d_path.as_posix())

    contact_fig = visualizer.plot_contact_map(contact_map)
    contact_path = output_dir / f"{slug}_contacts.html"
    contact_fig.write_html(contact_path.as_posix())

    print(" Visualizations saved:")
    print(f"   - {structure_path}")
    print(f"   - {structure_3d_path}")
    print(f"   - {contact_path}")


def generate_function_visuals(visualizer: SequenceVisualizer, output_dir: Path, summaries: List[Dict]) -> None:
    labels = [f"{summary['name']} ({summary['prediction']['function']})" for summary in summaries]
    sequences = [summary["sequence"] for summary in summaries]

    composition_fig = visualizer.plot_amino_acid_composition(sequences, labels)
    composition_path = output_dir / "protein_functions.html"
    composition_fig.write_html(composition_path.as_posix())

    print(" Function visualizations saved:")
    print(f"   - {composition_path}")


def main() -> None:
    args = parse_args()

    run_structure = args.mode in {"structure", "both"}
    run_function = args.mode in {"function", "both"}

    print("FoldAI - Protein Intelligence Platform")
    print("=" * 55)
    print(f" Mode: {args.mode.title()}")

    torch_module = None
    if run_function:
        try:
            import torch as torch_module  # type: ignore
        except ImportError as exc:
            raise ImportError("PyTorch is required for function prediction but is not installed.") from exc
        from analysis.protein_function_predictor import ProteinFunctionPredictorAI

        device = "cuda" if torch_module.cuda.is_available() else "cpu"
        print(f" Function predictor device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch_module.cuda.get_device_name(0)}")
    else:
        device = "cpu"
    print()

    uniprot = UniProtClient()
    sequences = gather_sequences(args, uniprot)

    if not sequences:
        print("No sequences supplied; falling back to demo proteins.")
        sequences = DEFAULT_DEMO

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = None if args.skip_visuals else SequenceVisualizer()
    structure_predictor = StructurePredictor() if run_structure else None
    if run_function:
        function_predictor = ProteinFunctionPredictorAI(device=device)
    else:
        function_predictor = None

    structure_summaries: List[Dict] = []
    function_summaries: List[Dict] = []

    print(f"Analyzing {len(sequences)} protein(s)...\n")

    for idx, (name, sequence) in enumerate(sequences, start=1):
        print(f"Protein {idx}: {name}")
        print(f"  Length: {len(sequence)} amino acids")

        if run_structure and structure_predictor:
            structure = structure_predictor.predict_secondary_structure(sequence)
            stability = structure_predictor.predict_protein_stability(sequence)
            aggregation = structure_predictor.predict_aggregation_propensity(sequence)
            print_structure_summary(structure, stability, aggregation)
            structure_summaries.append(
                {
                    "name": name,
                    "sequence": sequence,
                    "structure": structure,
                    "stability": stability,
                    "aggregation": aggregation,
                }
            )

        if run_function and function_predictor:
            prediction = function_predictor.predict_function(sequence)
            print_function_summary(prediction, args.top_k)
            function_summaries.append(
                {
                    "name": name,
                    "sequence": sequence,
                    "prediction": prediction,
                }
            )

        print()

    if run_structure and structure_summaries:
        primary_structure = structure_summaries[0]
        contact_map = structure_predictor.predict_contact_map(primary_structure["sequence"])
        domain_info = structure_predictor.identify_functional_domains(primary_structure["sequence"])
        if domain_info["domains"] or domain_info["motifs"]:
            print(f"Domain & motif insights for {primary_structure['name']}:")
            if domain_info["domains"]:
                top_domain = domain_info["domains"][0]
                print(f"  Domain detected: {top_domain['name']} ({top_domain['start']}-{top_domain['end']})")
            if domain_info["motifs"]:
                motif = domain_info["motifs"][0]
                print(f"  Motif example: {motif['name']} at position {motif['position']}")
            print()
        if visualizer:
            generate_structure_visuals(visualizer, output_dir, primary_structure, contact_map)
        print()

    if run_function and function_summaries:
        primary_function = function_summaries[0]
        analysis = function_predictor.analyze_functional_regions(primary_function["sequence"])
        print(f"Functional region analysis for {primary_function['name']}:")
        print(f"  Regions detected: {analysis['total_functional_regions']}")
        print(f"  Active sites: {len(analysis['active_sites'])}")
        if analysis["functional_regions"]:
            top_region = analysis["functional_regions"][0]
            print(f"  Top region ({top_region['start']}-{top_region['end']}): {top_region['sequence']}")
        print()

        print("Function explanations:")
        for summary in function_summaries[:3]:
            function = summary["prediction"]["function"]
            explanation = function_predictor.get_function_explanation(function)
            print(f" {function}: {explanation['description']}")
            print(f"  Examples: {explanation['examples']}")
            print(f"  Applications: {explanation['applications']}")
            print()

        if visualizer:
            generate_function_visuals(visualizer, output_dir, function_summaries)

    print("FoldAI run complete.")
    if not args.skip_visuals:
        print(f"Explore saved results in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
