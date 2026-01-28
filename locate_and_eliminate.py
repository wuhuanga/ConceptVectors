"""
Concept Elimination Pipeline:
1. Given a concept and a few example sentences, automatically locate concept vectors
   by comparing MLP activations between full sentences and masked sentences.
2. Extract top-k (layer, dim) pairs with largest activation differences.
3. Apply gradient-based unlearning on the located vectors.

Usage:
    python locate_and_eliminate.py \
        --model_path /path/to/llama2-7b \
        --concept "Harry Potter" \
        --sentences "Harry Potter is a fictional character created by J.K. Rowling." \
                    "Harry Potter attended Hogwarts School of Witchcraft and Wizardry." \
                    "Harry Potter is known as the Boy Who Lived." \
        --mask_token "something" \
        --top_k 5 \
        --forget_loss grad_ascent \
        --lr 0.2 \
        --num_epochs 1 \
        --batch_size 4 \
        --save_dir ./results
"""

import argparse
import torch
import json
import os
import re
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import set_random_seed


def get_mlp_output_hooks(model):
    """Register forward hooks on all MLP down_proj (value vector) layers
    to capture their output activations.

    Returns:
        hooks: list of hook handles (for removal)
        activations: dict mapping layer_index -> output tensor
    """
    activations = {}
    hooks = []

    if 'llama' in model.config.model_type:
        for layer_idx, layer in enumerate(model.model.layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output shape: (batch, seq_len, hidden_dim)
                    activations[idx] = output.detach()
                return hook_fn
            h = layer.mlp.down_proj.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    elif 'olmo' in model.config.model_type:
        for layer_idx, block in enumerate(model.model.transformer.blocks):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    activations[idx] = output.detach()
                return hook_fn
            h = block.ff_out.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    return hooks, activations


def get_mlp_input_hooks(model):
    """Register forward hooks on all MLP down_proj layers to capture their INPUT,
    which represents the intermediate MLP activations BEFORE the down projection.

    The down_proj weight is (hidden_dim, intermediate_dim), and its input has shape
    (batch, seq_len, intermediate_dim). Each column of down_proj.weight corresponds
    to one "dimension" in the concept vector framework.

    Returns:
        hooks: list of hook handles
        activations: dict mapping layer_index -> input tensor (batch, seq_len, intermediate_dim)
    """
    activations = {}
    hooks = []

    if 'llama' in model.config.model_type:
        for layer_idx, layer in enumerate(model.model.layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # input is a tuple, first element is the actual input tensor
                    activations[idx] = input[0].detach()
                return hook_fn
            h = layer.mlp.down_proj.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    elif 'olmo' in model.config.model_type:
        for layer_idx, block in enumerate(model.model.transformer.blocks):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    activations[idx] = input[0].detach()
                return hook_fn
            h = block.ff_out.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    return hooks, activations


def locate_concept_vectors(model, tokenizer, concept, sentences, mask_token="something", top_k=5):
    """Locate concept vectors by comparing MLP activations between
    original sentences and concept-masked sentences.

    For each sentence:
        - Original: "Harry Potter is a wizard."
        - Masked:   "something is a wizard."
    Feed both through the model, record the INPUT to each down_proj layer
    (i.e., the intermediate MLP activations). The dimensions with the largest
    average absolute activation difference across all sentences and tokens
    are the concept-encoding dimensions.

    Args:
        model: the language model
        tokenizer: tokenizer
        concept: concept string (e.g., "Harry Potter")
        sentences: list of example sentences containing the concept
        mask_token: replacement string for masking the concept
        top_k: number of (layer, dim) pairs to return

    Returns:
        list of (layer, dim, avg_diff) tuples sorted by avg_diff descending
    """
    model.eval()

    # Create masked sentences
    masked_sentences = []
    for sent in sentences:
        masked = sent.replace(concept, mask_token)
        if masked == sent:
            # Try case-insensitive replacement
            pattern = re.compile(re.escape(concept), re.IGNORECASE)
            masked = pattern.sub(mask_token, sent)
        masked_sentences.append(masked)

    print(f"\n=== Locating concept vectors for: '{concept}' ===")
    print(f"Number of example sentences: {len(sentences)}")
    for i, (orig, masked) in enumerate(zip(sentences, masked_sentences)):
        print(f"  [{i}] Original: {orig[:80]}...")
        print(f"       Masked:   {masked[:80]}...")

    # Accumulate differences across all sentences
    # diff_accum[layer_idx] will be a tensor of shape (intermediate_dim,)
    diff_accum = defaultdict(lambda: None)
    count = 0

    # Disable use_cache and convert to float32 to avoid autocast compatibility issues
    use_cache_orig = model.config.use_cache
    model.config.use_cache = False
    original_dtype = next(model.parameters()).dtype
    model.float()  # Convert to float32 for compatibility

    for orig_sent, mask_sent in zip(sentences, masked_sentences):
        # Forward pass on original sentence
        hooks_orig, acts_orig = get_mlp_input_hooks(model)
        inputs_orig = tokenizer(orig_sent, return_tensors="pt", padding=False).to(model.device)
        with torch.inference_mode():
            model(**inputs_orig)
        for h in hooks_orig:
            h.remove()

        # Forward pass on masked sentence
        hooks_mask, acts_mask = get_mlp_input_hooks(model)
        inputs_mask = tokenizer(mask_sent, return_tensors="pt", padding=False).to(model.device)
        with torch.inference_mode():
            model(**inputs_mask)
        for h in hooks_mask:
            h.remove()

        # Compute per-layer, per-dim activation difference
        for layer_idx in acts_orig:
            if layer_idx not in acts_mask:
                continue
            # Mean over batch and sequence dimensions -> (intermediate_dim,)
            orig_mean = acts_orig[layer_idx].abs().mean(dim=(0, 1))
            mask_mean = acts_mask[layer_idx].abs().mean(dim=(0, 1))
            diff = (orig_mean - mask_mean).abs()

            if diff_accum[layer_idx] is None:
                diff_accum[layer_idx] = diff
            else:
                diff_accum[layer_idx] = diff_accum[layer_idx] + diff

        count += 1

    # Restore model to original dtype and settings
    model.config.use_cache = use_cache_orig
    if original_dtype == torch.bfloat16:
        model.bfloat16()
    elif original_dtype == torch.float16:
        model.half()

    # Average and find top-k across all layers
    all_diffs = []
    for layer_idx in sorted(diff_accum.keys()):
        avg_diff = diff_accum[layer_idx] / count
        for dim_idx in range(avg_diff.shape[0]):
            all_diffs.append((layer_idx, dim_idx, avg_diff[dim_idx].item()))

    # Sort by difference descending
    all_diffs.sort(key=lambda x: x[2], reverse=True)
    top_vectors = all_diffs[:top_k]

    print(f"\n=== Top-{top_k} concept vectors ===")
    for rank, (layer, dim, diff_val) in enumerate(top_vectors):
        print(f"  #{rank+1}: Layer {layer}, Dim {dim}, AvgDiff = {diff_val:.6f}")

    return top_vectors


def add_noise_to_vectors(model, locations, noise_scale=0.1):
    """Add Gaussian noise to the located concept vector dimensions."""
    for layer, dim, _ in locations:
        if 'llama' in model.config.model_type:
            key = f'model.layers.{layer}.mlp.down_proj.weight'
        elif 'olmo' in model.config.model_type:
            key = f'model.transformer.blocks.{layer}.ff_out.weight'
        else:
            raise ValueError(f"Unsupported model type: {model.config.model_type}")

        param = model.state_dict()[key]
        shape = (param.shape[0],)
        noise = torch.normal(0, noise_scale, size=shape).to(param.device, dtype=param.dtype)
        param[:, dim] += noise
        print(f"  Added noise to Layer {layer}, Dim {dim}")


def setup_gradient_masks(model, locations):
    """Set up gradient masks so only the located concept vector dimensions are trained."""
    # Collect target (layer, dim) pairs
    target_map = defaultdict(list)
    for layer, dim, _ in locations:
        target_map[layer].append(dim)

    if 'llama' in model.config.model_type:
        for name, param in model.named_parameters():
            matched = False
            for layer_idx, dims in target_map.items():
                if f"layers.{layer_idx}.mlp.down_proj" in name:
                    param.requires_grad = True
                    mask = torch.zeros_like(param).to(param.device)
                    for d in dims:
                        mask[:, d] = 1
                    param.register_hook(lambda grad, m=mask: grad.mul_(m))
                    matched = True
                    print(f"  Trainable: {name} (dims: {dims})")
                    break
            if not matched:
                if "model.embed_tokens.weight" in name:
                    param.requires_grad = True
                    zero_mask = torch.zeros_like(param).to(param.device)
                    param.register_hook(lambda grad, m=zero_mask: grad.mul_(m))
                else:
                    param.requires_grad = False

    elif 'olmo' in model.config.model_type:
        for name, param in model.named_parameters():
            matched = False
            for layer_idx, dims in target_map.items():
                if f'blocks.{layer_idx}.ff_out.weight' in name:
                    param.requires_grad = True
                    mask = torch.zeros_like(param).to(param.device)
                    for d in dims:
                        mask[:, d] = 1
                    param.register_hook(lambda grad, m=mask: grad.mul_(m))
                    matched = True
                    print(f"  Trainable: {name} (dims: {dims})")
                    break
            if not matched:
                if "model.embed_tokens.weight" in name:
                    param.requires_grad = True
                    zero_mask = torch.zeros_like(param).to(param.device)
                    param.register_hook(lambda grad, m=zero_mask: grad.mul_(m))
                else:
                    param.requires_grad = False


def eliminate_concept(model, tokenizer, sentences, concept, locations,
                      forget_loss="grad_ascent", lr=0.2, num_epochs=1,
                      batch_size=4, gradient_accumulation_steps=8,
                      noise_scale=0.1, oracle_model=None,
                      beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, KL_coeff=1.0):
    """Run concept elimination training on the located vectors.

    Args:
        model: the model to modify
        tokenizer: tokenizer
        sentences: the example sentences (used as forget data)
        concept: concept string
        locations: list of (layer, dim, diff) from locate_concept_vectors
        forget_loss: loss type (grad_ascent, grad_diff, npo, etc.)
        lr: learning rate
        num_epochs: number of training epochs
        batch_size: batch size
        gradient_accumulation_steps: gradient accumulation steps
        noise_scale: scale of Gaussian noise to add before training
        oracle_model: reference model for NPO/DPO losses (None for grad_ascent/grad_diff)
    """
    import transformers
    from data_module import TextForgetDatasetWikipedia, split_paragraph
    from dataloader import CustomTrainerForgetting, custom_data_collator_forget
    from forget import EarlyStoppingCallback

    print(f"\n=== Eliminating concept: '{concept}' ===")
    print(f"  Loss type: {forget_loss}")
    print(f"  Locations: {[(l, d) for l, d, _ in locations]}")

    # Set up gradient masks for located vectors
    print("\nSetting up gradient masks...")
    setup_gradient_masks(model, locations)

    # Add noise to concept vectors
    print("\nAdding noise to concept vectors...")
    add_noise_to_vectors(model, locations, noise_scale=noise_scale)

    # Build forget dataset from the provided sentences
    content = " ".join(sentences)

    # Use a simple retain text (repeat sentence structure without concept)
    retain_sentences = []
    for s in sentences:
        masked = s.replace(concept, "something")
        retain_sentences.append(masked)
    random_content = [" ".join(retain_sentences)]

    # We need a temp data path for the dataset
    data_path = "/tmp/concept_eliminate_data"
    os.makedirs(data_path, exist_ok=True)

    # Determine model_family
    if 'llama' in model.config.model_type:
        model_family = 'llama2-7b'
    elif 'olmo' in model.config.model_type:
        model_family = 'olmo-7b'
    else:
        model_family = 'llama2-7b'

    max_length = 500
    torch_format_dataset = TextForgetDatasetWikipedia(
        data_path,
        content=content,
        random_content=random_content,
        tokenizer=tokenizer,
        model_family=model_family,
        max_length=max_length,
        split="wikipedia",
        loss_type=forget_loss
    )

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    steps_per_epoch = max(1, len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices))
    max_steps = int(num_epochs * len(torch_format_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    max_steps = max(max_steps, 1)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0,
        max_steps=max_steps,
        learning_rate=lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, steps_per_epoch),
        logging_dir=f'{data_path}/logs',
        output_dir=data_path,
        optim="paged_adamw_32bit",
        save_steps=max_steps + 1000000,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
    )

    if forget_loss in ['npo', 'npo_KL', 'npo_grad_diff', 'dpo', 'dpo_KL', 'dpo_grad_diff']:
        loss_threshold = 0
    else:
        loss_threshold = -100

    early_stopping_callback = EarlyStoppingCallback(loss_threshold=loss_threshold)

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model=oracle_model,
        forget_loss=forget_loss,
        seed=42,
        ref_policy="fine_tuned" if oracle_model else None,
        beta=beta,
        npo_coeff=npo_coeff,
        grad_diff_coeff=grad_diff_coeff,
        KL_coeff=KL_coeff,
        callbacks=[early_stopping_callback]
    )

    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    print("\n=== Concept elimination training complete ===")
    return model


def evaluate_elimination(model, tokenizer, concept, locations, questions=None, top_k=200):
    """Evaluate the concept elimination by checking vector projections and QA."""
    from forget import evaluate as eval_fn

    print(f"\n=== Evaluating elimination of '{concept}' ===")

    E = model.get_output_embeddings().weight.detach()

    results = []
    for layer, dim, diff_val in locations:
        if 'llama' in model.config.model_type:
            params = model.state_dict()[f'model.layers.{layer}.mlp.down_proj.weight'].T[dim, :]
        elif 'olmo' in model.config.model_type:
            params = model.state_dict()[f'model.transformer.blocks.{layer}.ff_out.weight'].T[dim, :]

        logits = params.T.matmul(E.T)
        _, sorted_indices = torch.sort(logits, descending=True)
        ids = [i.item() for i in sorted_indices[:top_k]]
        projection = [tokenizer._convert_id_to_token(i) for i in ids]
        results.append({
            'layer': layer,
            'dim': dim,
            'top_tokens': projection[:20],  # show top 20
        })
        print(f"  Layer {layer}, Dim {dim} top-20 tokens: {projection[:20]}")

    # If questions provided, test generation
    if questions:
        qa_prompts = [f"Question: {q}\n Answer:" for q in questions]
        inputs = tokenizer(qa_prompts, return_tensors="pt", padding=True,
                          return_token_type_ids=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=100)
        outputs = tokenizer.batch_decode(gen[:, -100:], skip_special_tokens=True)
        print(f"\n  QA Evaluation:")
        for q, a in zip(questions, outputs):
            print(f"    Q: {q}")
            print(f"    A: {a[:100]}")
        results.append({'qa_answers': outputs})

    return results


def main():
    parser = argparse.ArgumentParser(description="Locate and eliminate concept vectors")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--concept", type=str, required=True, help="Concept to eliminate (e.g., 'Harry Potter')")
    parser.add_argument("--sentences", type=str, nargs="+", required=True,
                        help="Example sentences containing the concept")
    parser.add_argument("--mask_token", type=str, default="something",
                        help="Token to replace concept with for masking")
    parser.add_argument("--top_k", type=int, default=5, help="Number of concept vectors to locate")
    parser.add_argument("--forget_loss", type=str, default="grad_ascent",
                        choices=["grad_ascent", "grad_diff", "npo", "npo_KL", "npo_grad_diff", "dpo"],
                        help="Unlearning loss type")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Noise scale for initialization")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--questions", type=str, nargs="*", default=None,
                        help="Optional QA questions to evaluate after elimination")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locate_only", action="store_true",
                        help="Only locate concept vectors, do not eliminate")

    args = parser.parse_args()
    set_random_seed(args.seed)

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    # Step 1: Locate concept vectors
    locations = locate_concept_vectors(
        model, tokenizer, args.concept, args.sentences,
        mask_token=args.mask_token, top_k=args.top_k
    )

    # Save located vectors
    os.makedirs(args.save_dir, exist_ok=True)
    loc_path = os.path.join(args.save_dir, f"located_vectors_{args.concept.replace(' ', '_')}.json")
    with open(loc_path, "w") as f:
        json.dump([{"layer": l, "dim": d, "diff": v} for l, d, v in locations], f, indent=2)
    print(f"\nSaved located vectors to {loc_path}")

    if args.locate_only:
        print("locate_only mode, skipping elimination.")
        return

    # Step 2 & 3: Eliminate concept
    # Load oracle model if needed for NPO/DPO
    oracle_model = None
    if args.forget_loss not in ['grad_ascent', 'grad_diff']:
        oracle_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()

    model = eliminate_concept(
        model, tokenizer, args.sentences, args.concept, locations,
        forget_loss=args.forget_loss, lr=args.lr,
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        noise_scale=args.noise_scale, oracle_model=oracle_model,
    )

    # Evaluate
    eval_results = evaluate_elimination(
        model, tokenizer, args.concept, locations, questions=args.questions
    )

    # Save results
    results_path = os.path.join(args.save_dir, f"elimination_results_{args.concept.replace(' ', '_')}.pt")
    torch.save({
        "concept": args.concept,
        "locations": [(l, d, v) for l, d, v in locations],
        "eval_results": eval_results,
    }, results_path)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
