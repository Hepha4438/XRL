import argparse
import json
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Prune unused actions from trained SAE Logic V3 model.")
    parser.add_argument("--save_dir", type=str, default="./sae_logic_v3_outputs", help="Directory containing training_data.pt, sae_logic_v3_model.pt, and learned_rules.json")
    args = parser.parse_args()

    data_path = os.path.join(args.save_dir, "training_data.pt")
    model_path = os.path.join(args.save_dir, "sae_logic_v3_model.pt")
    json_path = os.path.join(args.save_dir, "learned_rules.json")

    if not os.path.exists(data_path):
        print(f"Error: Could not find data file {data_path}")
        return

    print(f"Loading training data from: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    used_actions = torch.unique(data['actions']).tolist()
    print(f"Used action indices: {used_actions}")

    print(f"Loading model checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    state_dict = checkpoint['model_state']
    config_dict = checkpoint['config']
    
    n_actions = config_dict.get('n_actions', 7)
    n_clauses = config_dict.get('n_clauses_per_action', 10)
    
    unused_indices = [i for i in range(n_actions) if i not in used_actions]
    print(f"Unused action indices to prune: {unused_indices}")

    if not unused_indices:
        print("No unused actions found. Exiting.")
        return

    # Clone weights to update
    cw = state_dict['logic_layer.clause_weight'].clone()
    w_pos = state_dict['logic_layer.w_pos'].clone()
    w_neg = state_dict['logic_layer.w_neg'].clone()
    
    for idx in unused_indices:
        start = idx * n_clauses
        end = (idx + 1) * n_clauses
        
        # Push clause weights down so Sigmoid(cb) -> 0.0
        cw[start:end] = -100.0
        
        # Push selector weights (w_pos, w_neg) down so absent logit (0.0) gets 100% prob
        w_pos[start:end, :] = -100.0
        w_neg[start:end, :] = -100.0

    # Save back to state_dict
    state_dict['logic_layer.clause_weight'] = cw
    state_dict['logic_layer.w_pos'] = w_pos
    state_dict['logic_layer.w_neg'] = w_neg
    checkpoint['model_state'] = state_dict

    # Load JSON to get action names mapping and update it
    with open(json_path, 'r') as f:
        rules_json = json.load(f)
        
    action_names = list(rules_json.keys())
    
    for idx in unused_indices:
        if idx < len(action_names):
            act_name = action_names[idx]
            print(f"Pruning rules for action: {act_name}")
            rules_json[act_name] = [] # Empty rules
            if 'rules' in checkpoint and act_name in checkpoint['rules']:
                checkpoint['rules'][act_name] = []

    print(f"Saving patched model to {model_path}")
    torch.save(checkpoint, model_path)
    
    print(f"Saving patched JSON to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(rules_json, f, indent=2, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    main()
