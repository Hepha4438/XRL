#!/usr/bin/env python3
"""
gemini_concept_labeler.py
Uses Gemini API (via the new google.genai SDK) to read IG Heatmap summary images 
and automatically assign semantic labels based on Contrastive Visual + Logical Context.
Evaluates "monoscore" to enforce monosemantic labels and handles environmental contexts.
"""

import os
import glob
import json
import argparse
import re
from PIL import Image
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Please install the new Gemini SDK: pip install google-genai")

def setup_gemini(api_key):
    client = genai.Client(api_key=api_key)
    return client

def load_and_deduplicate_rules(rules_path):
    if not rules_path or not os.path.exists(rules_path):
        return None
    with open(rules_path, 'r', encoding='utf-8') as f:
        raw_rules = json.load(f)
    deduped_rules = {}
    for action, clauses in raw_rules.items():
        if not clauses or clauses[0] == "(no active clauses)": continue
        clean_clauses = set()
        for clause in clauses:
            clean_clause = re.sub(r'\s*\[bias=[^\]]+\]', '', clause)
            clean_clauses.add(clean_clause)
        if clean_clauses: deduped_rules[action] = list(clean_clauses)
    return deduped_rules

def get_rules_for_concept(concept_id, deduped_rules):
    if not deduped_rules: return ""
    num_str = ''.join(filter(str.isdigit, concept_id))
    if not num_str: return ""
    relevant_rules = []
    pattern = re.compile(fr'f_{num_str}\b')
    for action, clauses in deduped_rules.items():
        for clause in clauses:
            if pattern.search(clause):
                relevant_rules.append(f"IF {clause} THEN {action}")
    if not relevant_rules: return ""
    return "\n".join([f"- {rule}" for rule in relevant_rules])

def get_environment_context(env_name):
    """Provides specific context for the 3 target environments."""
    contexts = {
        "MiniGrid-Dynamic-Obstacles-5x5-v0": (
            "This is a 5x5 grid world. The agent (red triangle) must navigate to the green goal square "
            "while avoiding moving blue obstacle balls. The agent's ego-centric view is a local 7x7 grid."
        ),
        "MiniGrid-DoorKey-6x6-v0": (
            "This is a 6x6 grid world. The agent (red triangle) must pick up a yellow key, use it to unlock "
            "a yellow door, and navigate to the green goal square. The agent's ego-centric view is a local 7x7 grid."
        ),
        "PixelCartPole-v0": (
            "This is a pixel-based CartPole environment. A pole is attached by an un-actuated joint to a cart, "
            "which moves along a frictionless track. The agent applies forces (Left/Right) to the cart to keep "
            "the pole balanced upright. The input is a raw pixel render of the cart and pole."
        )
    }
    return contexts.get(env_name, f"The agent is playing the environment: {env_name}.")

def build_prompt(env_name, top_k, rules_context=""):
    env_context = get_environment_context(env_name)

    prompt = f"""You are an expert AI researcher analyzing Concept Neurons in a Deep Reinforcement Learning agent.

=========================================
**ENVIRONMENT CONTEXT:**
{env_context}
=========================================

I am providing an image containing the Top {top_k} scenarios for a single concept neuron. 
The image uses a CONTRASTIVE layout. Each row directly compares an ACTIVATED state (ON) vs an INACTIVE state (OFF) for this concept. 
There are 6 panels from left to right in each row:

--- ACTIVATED EXAMPLES (Panels 1-3) ---
1. Panel 1 (labeled [God View (ON)] or [Full Render (ON)]): The global, uncropped view of the game environment.
2. Panel 2 (labeled [Agent View (ON)] or [Model Input (ON)]): The agent's actual visual observation. For MiniGrid environments, this is a local 7x7 grid where the agent is ALWAYS assumed to be at the bottom-center looking UP. It is NOT rotated to align with the global God View.
3. Panel 3 (labeled [IG Heatmap (ON)]): The Integrated Gradients attribution map. This map aligns perfectly with Panel 2. Red/Jet areas show EXACTLY which pixels TRIGGERED the concept.

--- INACTIVE EXAMPLES (Panels 4-6) ---
4. Panel 4 (labeled [God View (OFF)] or [Full Render (OFF)]): The global view when the concept is NOT activated.
5. Panel 5 (labeled [Agent View (OFF)] or [Model Input (OFF)]): The agent's observation when the concept is OFF.
6. Panel 6 (labeled [IG Heatmap (OFF)]): The attribution map when deactivated.
"""

    if rules_context:
        prompt += f"""
=========================================
**LOGICAL CONTEXT (CRITICAL HINT):**
The AI agent explicitly uses this concept to make decisions. Here are the deduplicated DNF rules involving this concept:
{rules_context}

*Hint: Use these rules to guide your visual search. If the concept triggers a 'Pickup' action, look for an object to pick up in the ON heatmaps.*
=========================================
"""

    prompt += """
Your task:
1. Perform Contrastive Analysis: Compare the ON panels with the OFF panels across ALL rows. What semantic feature is present in ON but missing in OFF? Look at the Red/Jet regions in the IG Heatmap.
2. Evaluate Monosemanticity (Consistency vs Exceptions): Look closely at every single ON row. Does the concept trigger on exactly ONE clear semantic feature every single time? Or are there exceptions? (e.g., 7 rows show a key, but 1 row shows a wall/empty space).
3. Assign a 'monoscore' from 0 to 5 evaluating how monosemantic the concept is:
   - 5: Perfectly monosemantic (100% consistent across ALL ON panels).
   - 3-4: Mostly monosemantic, but with 1 or 2 clear exceptions/deviations.
   - 1-2: Polysemantic (triggers on multiple distinct, unrelated features).
   - 0: Completely uninterpretable or random.
4. Generate a concise, descriptive snake_case label based on the visual and logical evidence. Provide an OBJECTIVE assessment:
   - If the concept is monosemantic, give it a specific, single-feature label. Do NOT use "super concept" labels (OR-ing unrelated things like "key_or_wall") if it clearly represents one main idea.
   - Use hedging prefixes like "likely_", "mostly_", or "partially_" ONLY if a single semantic feature dominates the majority of the ON images but is missing in a few exceptions.
   - If the concept is objectively polysemantic (triggers on completely different things), reflect the objective reality in the label based on what it actually captures (even if it results in a compound label).
5. Assign a confidence score from 1 to 5 for your overall assessment.
6. Provide reasoning explaining the contrast, the heatmaps, AND explicitly mentioning any exceptions that affected the monoscore.

Output EXACTLY in this JSON format:
{
  "concept_id": "C_ID_FROM_TITLE",
  "label": "your_snake_case_label",
  "monoscore": 4,
  "confidence": 4,
  "reasoning": "Explain the contrast and explicitly note any exceptions across the rows."
}
"""
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing C*_summary.png images")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name for context")
    parser.add_argument("--top_k", type=int, default=8, help="Number of Top-K frames used")
    parser.add_argument("--rules_path", type=str, default="", help="Path to learned_rules.json for logical context")
    parser.add_argument("--api_key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="Google Gemini API Key")
    parser.add_argument("--model_name", type=str, default="gemini-3-flash-preview", help="Gemini API model version to use")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Please provide an API Key via --api_key or GEMINI_API_KEY environment variable")

    client = setup_gemini(args.api_key)
    deduped_rules = load_and_deduplicate_rules(args.rules_path)
    
    if deduped_rules:
        print(f"[+] Successfully loaded and deduplicated rules from {args.rules_path}")

    img_paths = glob.glob(os.path.join(args.img_dir, "C*_summary.png"))
    img_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0][1:])) 
    
    if not img_paths:
        print(f"No C*_summary.png images found in {args.img_dir}!")
        return

    print(f"\n[+] Found {len(img_paths)} concepts requiring labeling. Calling {args.model_name} API...")
    
    results = {
        "environment": args.env_name,
        "model_used": args.model_name,
        "total_used_concepts": len(img_paths),
        "concepts": {}
    }

    for img_path in tqdm(img_paths, desc="Labeling Concepts"):
        filename = os.path.basename(img_path)
        concept_id = filename.split('_')[0] 
        
        rules_context = get_rules_for_concept(concept_id, deduped_rules)
        prompt = build_prompt(args.env_name, args.top_k, rules_context)
        
        try:
            img = Image.open(img_path)
            response = client.models.generate_content(
                # Truyền tham số model_name vào đây
                model=args.model_name,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            result_json = json.loads(response.text)
            result_json["concept_id"] = concept_id
            
            results["concepts"][concept_id] = {
                "label": result_json.get("label", "unknown_concept"),
                "monoscore": result_json.get("monoscore", 0),
                "confidence": result_json.get("confidence", 0),
                "reasoning": result_json.get("reasoning", "")
            }
            
        except Exception as e:
            print(f"\nError labeling {concept_id}: {e}")
            results["concepts"][concept_id] = {
                "label": f"error_{concept_id}",
                "monoscore": 0,
                "confidence": 0,
                "reasoning": f"API Error: {str(e)}"
            }

    out_json_path = os.path.join(args.img_dir, "concept_labels.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n✓ Complete! Grounded concept labels saved to: {out_json_path}")
    print(f"Model used: {args.model_name}")
    print("Preview of labeled Concepts:")
    
    preview_count = min(5, len(results["concepts"]))
    for i, (cid, data) in enumerate(results["concepts"].items()):
        if i >= preview_count: break
        print(f"  - {cid}: {data['label']} (Mono: {data['monoscore']}/5, Conf: {data['confidence']}/5)")

if __name__ == "__main__":
    main()