"""
SVD Analysis for Unlearning Methods

This script analyzes how different unlearning methods affect the singular value
decomposition of weight matrices (Attention and MLP layers).

Research Questions:
1. Do different unlearning methods produce characteristic SVD changes?
2. Can we identify specific singular vector directions associated with concepts?
3. Can targeted removal of singular directions achieve precise forgetting?

Usage:
    python svd_analysis.py \
        --model_path /path/to/llama2-7b \
        --concept "Harry Potter" \
        --sentences "Harry Potter is a wizard." "Harry Potter attended Hogwarts." \
        --methods grad_ascent grad_diff npo npo_KL \
        --save_dir ./svd_results
"""

import argparse
import torch
import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import set_random_seed

# Monkey-patch for PyTorch/transformers compatibility
_original_is_autocast_enabled = torch.is_autocast_enabled
def _patched_is_autocast_enabled(device_type=None):
    try:
        if device_type is not None:
            return _original_is_autocast_enabled(device_type)
        else:
            return _original_is_autocast_enabled()
    except TypeError:
        return _original_is_autocast_enabled()
torch.is_autocast_enabled = _patched_is_autocast_enabled


def get_weight_matrices(model, model_type='llama'):
    """Extract all relevant weight matrices from the model.

    Returns dict with keys like:
        'layer.0.attn.q_proj': tensor
        'layer.0.attn.k_proj': tensor
        'layer.0.attn.v_proj': tensor
        'layer.0.attn.o_proj': tensor
        'layer.0.mlp.gate_proj': tensor
        'layer.0.mlp.up_proj': tensor
        'layer.0.mlp.down_proj': tensor
    """
    weights = {}

    if 'llama' in model_type:
        for layer_idx, layer in enumerate(model.model.layers):
            # Attention weights
            weights[f'layer.{layer_idx}.attn.q_proj'] = layer.self_attn.q_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.attn.k_proj'] = layer.self_attn.k_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.attn.v_proj'] = layer.self_attn.v_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.attn.o_proj'] = layer.self_attn.o_proj.weight.detach().cpu().float()
            # MLP weights
            weights[f'layer.{layer_idx}.mlp.gate_proj'] = layer.mlp.gate_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.mlp.up_proj'] = layer.mlp.up_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.mlp.down_proj'] = layer.mlp.down_proj.weight.detach().cpu().float()

    elif 'olmo' in model_type:
        for layer_idx, block in enumerate(model.model.transformer.blocks):
            weights[f'layer.{layer_idx}.attn.q_proj'] = block.att_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.mlp.ff_proj'] = block.ff_proj.weight.detach().cpu().float()
            weights[f'layer.{layer_idx}.mlp.ff_out'] = block.ff_out.weight.detach().cpu().float()

    return weights


def compute_svd(weight_matrix, top_k=None):
    """Compute SVD of a weight matrix.

    W = U @ diag(S) @ V^T

    Returns:
        U: left singular vectors (columns)
        S: singular values (descending)
        Vt: right singular vectors (rows)
    """
    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
    if top_k is not None:
        U = U[:, :top_k]
        S = S[:top_k]
        Vt = Vt[:top_k, :]
    return U, S, Vt


def compare_svd(weights_before, weights_after, top_k=100):
    """Compare SVD before and after unlearning.

    Returns dict with analysis results for each weight matrix.
    """
    results = {}

    for name in weights_before:
        if name not in weights_after:
            continue

        W_before = weights_before[name]
        W_after = weights_after[name]

        # Compute SVD
        U_b, S_b, Vt_b = compute_svd(W_before, top_k=top_k)
        U_a, S_a, Vt_a = compute_svd(W_after, top_k=top_k)

        # 1. Singular value changes
        sv_diff = (S_a - S_b).numpy()
        sv_rel_diff = ((S_a - S_b) / (S_b + 1e-8)).numpy()

        # 2. Singular vector alignment (cosine similarity)
        # For each singular vector, how much did it rotate?
        U_alignment = torch.abs(torch.sum(U_b * U_a, dim=0)).numpy()  # |u_before · u_after|
        V_alignment = torch.abs(torch.sum(Vt_b * Vt_a, dim=1)).numpy()  # |v_before · v_after|

        # 3. Weight difference projection onto singular directions
        W_diff = W_after - W_before
        # Project diff onto each right singular vector
        diff_proj_V = torch.abs(W_diff @ Vt_b.T).mean(dim=0).numpy()  # contribution of each V direction
        diff_proj_U = torch.abs(U_b.T @ W_diff).mean(dim=1).numpy()   # contribution of each U direction

        # 4. Find most changed directions
        U_change_idx = np.argsort(1 - U_alignment)[:10]  # top 10 most rotated U vectors
        V_change_idx = np.argsort(1 - V_alignment)[:10]  # top 10 most rotated V vectors

        results[name] = {
            'singular_values_before': S_b.numpy(),
            'singular_values_after': S_a.numpy(),
            'sv_diff': sv_diff,
            'sv_rel_diff': sv_rel_diff,
            'U_alignment': U_alignment,
            'V_alignment': V_alignment,
            'diff_proj_V': diff_proj_V,
            'diff_proj_U': diff_proj_U,
            'most_changed_U_idx': U_change_idx,
            'most_changed_V_idx': V_change_idx,
            'frobenius_norm_diff': torch.norm(W_diff, 'fro').item(),
        }

    return results


def run_unlearning(model, tokenizer, concept, sentences, method,
                   lr=0.2, num_epochs=1, batch_size=4, oracle_model=None,
                   use_bf16=True):
    """Run a specific unlearning method and return the modified model."""
    from locate_and_eliminate import locate_concept_vectors, eliminate_concept

    # First locate concept vectors
    locations = locate_concept_vectors(
        model, tokenizer, concept, sentences,
        mask_token="something", top_k=5
    )

    # Run elimination
    model = eliminate_concept(
        model, tokenizer, sentences, concept, locations,
        forget_loss=method, lr=lr, num_epochs=num_epochs,
        batch_size=batch_size, oracle_model=oracle_model,
        use_bf16=use_bf16,
    )

    return model, locations


def analyze_singular_direction_removal(model, tokenizer, concept, sentences,
                                       weights_before, svd_results,
                                       target_layers=None, top_directions=5):
    """
    Experimental: Try removing specific singular directions to achieve forgetting.

    Hypothesis: If concept is encoded in specific singular directions,
    zeroing those directions might remove the concept while preserving utility.
    """
    from locate_and_eliminate import locate_concept_vectors

    results = {}

    # Focus on MLP down_proj layers (where concept vectors are typically found)
    if target_layers is None:
        target_layers = [name for name in svd_results if 'mlp.down_proj' in name]

    for layer_name in target_layers[:3]:  # Test on first 3 layers
        svd_info = svd_results[layer_name]

        # Get the most changed V directions (right singular vectors)
        changed_idx = svd_info['most_changed_V_idx'][:top_directions]

        # Get original weight
        W_orig = weights_before[layer_name]
        U, S, Vt = compute_svd(W_orig)

        # Zero out the most changed directions
        S_modified = S.clone()
        S_modified[changed_idx] = 0

        # Reconstruct weight
        W_modified = U @ torch.diag(S_modified) @ Vt

        results[layer_name] = {
            'removed_directions': changed_idx.tolist(),
            'original_sv': S[changed_idx].numpy().tolist(),
            'weight_diff_norm': torch.norm(W_orig - W_modified, 'fro').item(),
        }

        print(f"\n{layer_name}:")
        print(f"  Removed directions: {changed_idx}")
        print(f"  Original singular values at those positions: {S[changed_idx].numpy()}")
        print(f"  Weight change norm: {results[layer_name]['weight_diff_norm']:.4f}")

    return results


def plot_svd_analysis(svd_results, method_name, save_dir):
    """Generate visualization of SVD changes."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Singular value changes across layers (for MLP down_proj)
    mlp_layers = [k for k in svd_results if 'mlp.down_proj' in k]
    mlp_layers = sorted(mlp_layers, key=lambda x: int(x.split('.')[1]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SVD Analysis: {method_name}', fontsize=14)

    # Plot 1: Top-50 singular value changes
    ax = axes[0, 0]
    for i, layer in enumerate(mlp_layers[::4]):  # every 4th layer
        sv_diff = svd_results[layer]['sv_rel_diff'][:50]
        ax.plot(sv_diff, label=layer.split('.')[1], alpha=0.7)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Relative Change')
    ax.set_title('Singular Value Changes (top 50)')
    ax.legend(title='Layer', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: U vector alignment
    ax = axes[0, 1]
    for i, layer in enumerate(mlp_layers[::4]):
        alignment = svd_results[layer]['U_alignment'][:50]
        ax.plot(alignment, label=layer.split('.')[1], alpha=0.7)
    ax.set_xlabel('Singular Vector Index')
    ax.set_ylabel('Alignment (cosine)')
    ax.set_title('Left Singular Vector Alignment')
    ax.legend(title='Layer', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: V vector alignment
    ax = axes[1, 0]
    for i, layer in enumerate(mlp_layers[::4]):
        alignment = svd_results[layer]['V_alignment'][:50]
        ax.plot(alignment, label=layer.split('.')[1], alpha=0.7)
    ax.set_xlabel('Singular Vector Index')
    ax.set_ylabel('Alignment (cosine)')
    ax.set_title('Right Singular Vector Alignment')
    ax.legend(title='Layer', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Weight diff projection onto V directions
    ax = axes[1, 1]
    for i, layer in enumerate(mlp_layers[::4]):
        proj = svd_results[layer]['diff_proj_V'][:50]
        ax.plot(proj, label=layer.split('.')[1], alpha=0.7)
    ax.set_xlabel('Singular Direction Index')
    ax.set_ylabel('Projection Magnitude')
    ax.set_title('Weight Diff Projection onto V')
    ax.legend(title='Layer', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'svd_analysis_{method_name}.png'), dpi=150)
    plt.close()

    # Plot 5: Heatmap of Frobenius norm changes across layers
    fig, ax = plt.subplots(figsize=(12, 6))

    layer_types = ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj',
                   'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    num_layers = len([k for k in svd_results if 'layer.0.' in k]) // len([lt for lt in layer_types if any(lt in k for k in svd_results)])

    # Build heatmap data
    heatmap_data = []
    valid_types = []
    for lt in layer_types:
        row = []
        for layer_idx in range(num_layers):
            key = f'layer.{layer_idx}.{lt}'
            if key in svd_results:
                row.append(svd_results[key]['frobenius_norm_diff'])
        if row:
            heatmap_data.append(row)
            valid_types.append(lt)

    if heatmap_data:
        heatmap_data = np.array(heatmap_data)
        im = ax.imshow(heatmap_data, aspect='auto', cmap='hot')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Weight Type')
        ax.set_yticks(range(len(valid_types)))
        ax.set_yticklabels(valid_types)
        ax.set_title(f'Weight Change Magnitude (Frobenius Norm) - {method_name}')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'weight_changes_{method_name}.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SVD Analysis for Unlearning")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--sentences", type=str, nargs="+", required=True)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["grad_ascent", "grad_diff", "npo", "npo_KL"],
                        choices=["grad_ascent", "grad_diff", "npo", "npo_grad_diff", "npo_KL", "dpo"])
    parser.add_argument("--save_dir", type=str, default="./svd_results")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k_svd", type=int, default=100,
                        help="Number of top singular values to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on (cuda or cpu)")

    args = parser.parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Determine device and dtype
    if args.device == "cpu":
        device = torch.device("cpu")
        dtype = torch.float32
        print("Running on CPU (this will be slow but works without compatible GPU)")
    else:
        device = torch.device("cuda")
        dtype = torch.bfloat16

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Determine model type
    if 'llama' in args.model_path.lower():
        model_type = 'llama'
    elif 'olmo' in args.model_path.lower():
        model_type = 'olmo'
    else:
        model_type = 'llama'  # default

    all_results = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Running method: {method}")
        print(f"{'='*60}")

        # Load fresh model for each method
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)

        # Get weights before unlearning
        print("Extracting weights before unlearning...")
        weights_before = get_weight_matrices(model, model_type)

        # Load oracle model for NPO/DPO methods
        oracle_model = None
        if method not in ['grad_ascent', 'grad_diff']:
            print("Loading oracle model...")
            oracle_model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, trust_remote_code=True
            ).to(device)

        # Run unlearning
        print(f"Running {method} unlearning...")
        use_bf16 = (args.device == "cuda")
        try:
            model, locations = run_unlearning(
                model, tokenizer, args.concept, args.sentences, method,
                lr=args.lr, num_epochs=args.num_epochs,
                batch_size=args.batch_size, oracle_model=oracle_model,
                use_bf16=use_bf16
            )
        except Exception as e:
            print(f"Error running {method}: {e}")
            continue

        # Get weights after unlearning
        print("Extracting weights after unlearning...")
        weights_after = get_weight_matrices(model, model_type)

        # Compare SVD
        print("Computing SVD analysis...")
        svd_results = compare_svd(weights_before, weights_after, top_k=args.top_k_svd)

        # Generate plots
        print("Generating visualizations...")
        plot_svd_analysis(svd_results, method, args.save_dir)

        # Analyze singular direction removal
        print("Analyzing singular direction removal...")
        removal_results = analyze_singular_direction_removal(
            model, tokenizer, args.concept, args.sentences,
            weights_before, svd_results
        )

        # Store results
        all_results[method] = {
            'concept_locations': [(l, d, v) for l, d, v in locations],
            'svd_summary': {
                name: {
                    'frobenius_norm_diff': info['frobenius_norm_diff'],
                    'most_changed_U_idx': info['most_changed_U_idx'].tolist(),
                    'most_changed_V_idx': info['most_changed_V_idx'].tolist(),
                    'mean_sv_rel_diff': float(np.mean(np.abs(info['sv_rel_diff'][:50]))),
                    'mean_U_alignment': float(np.mean(info['U_alignment'][:50])),
                    'mean_V_alignment': float(np.mean(info['V_alignment'][:50])),
                }
                for name, info in svd_results.items()
            },
            'removal_analysis': removal_results,
        }

        # Clean up
        del model
        if oracle_model is not None:
            del oracle_model
        torch.cuda.empty_cache()

    # Save all results
    results_path = os.path.join(args.save_dir, 'svd_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Method Comparison")
    print("="*60)

    for method, results in all_results.items():
        print(f"\n{method}:")

        # Average changes in MLP down_proj
        mlp_keys = [k for k in results['svd_summary'] if 'mlp.down_proj' in k]
        if mlp_keys:
            avg_frob = np.mean([results['svd_summary'][k]['frobenius_norm_diff'] for k in mlp_keys])
            avg_sv_diff = np.mean([results['svd_summary'][k]['mean_sv_rel_diff'] for k in mlp_keys])
            avg_U_align = np.mean([results['svd_summary'][k]['mean_U_alignment'] for k in mlp_keys])
            avg_V_align = np.mean([results['svd_summary'][k]['mean_V_alignment'] for k in mlp_keys])

            print(f"  MLP down_proj (avg across layers):")
            print(f"    Frobenius norm diff: {avg_frob:.4f}")
            print(f"    Mean SV relative diff: {avg_sv_diff:.4f}")
            print(f"    Mean U alignment: {avg_U_align:.4f}")
            print(f"    Mean V alignment: {avg_V_align:.4f}")


if __name__ == "__main__":
    main()
