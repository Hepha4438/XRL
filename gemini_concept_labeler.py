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
    is_minigrid = 'MiniGrid' in env_name

    # --- DYNAMIC RULES BASED ON ENVIRONMENT ---
    if is_minigrid:
        visual_rules = f"""
1. HOLISTIC SYNTHESIS (God View + Agent View + Rules): Do NOT overly fixate on just one panel or one single row. Synthesize the big picture to find the COMMON DENOMINATOR! Look at the [God View] to understand the agent's overall situation (e.g., where is the key relative to the agent?). Use the [IG Heatmap] to see its local attention. Look at the [Logical Context] for the intended Action.
2. THE "TURN" ANOMALY (CRITICAL DOMAIN KNOWLEDGE): Check the provided Logical Context. If the agent only has rules for ONE turning direction (e.g., 'TurnRight' but no 'TurnLeft', or vice versa), it means the agent uses that single action to CHANGE DIRECTION entirely. Therefore, do NOT automatically assume a turn means "obstacle ahead". It often triggers simply because a target (like a key) is off-center or behind the agent.
3. SPATIAL MAPPING VIA RED BOX: The **RED BOX** in the [Agent View] represents the agent.
   - CARRYING A KEY: Only if the yellow key is drawn strictly INSIDE the RED BOX.
   - IN FRONT: Only if the object is in the cell strictly ABOVE the RED BOX.
   - OFF-CENTER OR BEHIND: If the object is to the left, right, or in the bottom corners of the Agent View.
4. HEATMAP AS A FOCAL POINT, NOT A BLINDER: The DARK RED spots show local attention. Trace them to the Agent View relative to the RED BOX. However, do NOT get tunnel vision. You must interpret what that highlighted cell means within the broader context of the [God View]. If the Heatmap says "Zero Attrib.", prioritize the holistic state (e.g., `likely_key_off_center`). 
5. RL FUNCTIONAL SEMANTICS: If the concept triggers on visually different objects (e.g., an empty floor cell AND a door) that serve the SAME functional purpose (e.g., both allow moving 'Forward'), label it functionally (e.g., `traversable_path_ahead`).
6. NAMING STRATEGY (CRITICAL): Your label MUST perfectly bridge the visual details across ALL {top_k} ON cases with the logical Action. Ask yourself: "What specific state is consistently present in ALL these images that makes the agent logically want to perform THIS specific action?". Do not name it based on a single outlier row.
7. HOLISTIC FULL SCAN: You MUST scan ALL {top_k} ON cases from top to bottom before concluding.
"""
        red_box_notice = "**Notice the RED BOX drawn at the bottom-center: this permanently marks the agent's exact location.**"
        task_mapping_instruction = "relative to the RED BOX"
        task_reasoning_extra = "- Where the objects/attention are located RELATIVE to the RED BOX.\n   - How you formulated the label to act as the COMMON DENOMINATOR across all rows and the intended Action."
    else:
        visual_rules = f"""
1. HOLISTIC SYNTHESIS (Full Render + Model Input + Rules): Do NOT overly fixate on just one panel or one single row. Synthesize the big picture to find the COMMON DENOMINATOR! Look at the [Full Render (ON)] to understand the global physical state. Use the [IG Heatmap] to pinpoint exactly which physical parts the network is attending to. Look at the [Logical Context] for the intended Action.
2. 1-to-1 SPATIAL MAPPING (STRICT): Find the DARK RED pixels in the Heatmap. Trace them perfectly to the same location in the [Model Input (ON)] to identify EXACTLY what object, boundary, or state is located there. DO NOT hallucinate.
3. HEATMAP AS A FOCAL POINT, NOT A BLINDER: The DARK RED spots show exactly where the network is looking, but you must interpret THAT spot within the broader context of the [Full Render (ON)]. If a Heatmap says "Zero Attrib.", base your conclusion STRICTLY on the rows that actually have clear DARK RED spots, while triangulating with the full render.
4. RL FUNCTIONAL SEMANTICS: If the concept triggers on visually varying states that serve the SAME functional purpose for the action (e.g., different pole angles that both require moving the cart left to balance), label it functionally (e.g., `pole_falling_right`).
5. NAMING STRATEGY (CRITICAL): Your label MUST perfectly bridge the visual details across ALL {top_k} ON cases with the logical Action. Ask yourself: "What specific state is consistently present in ALL these images that makes the agent logically want to perform THIS specific action?". Do not name it based on a single outlier row.
6. HOLISTIC FULL SCAN: You MUST scan ALL {top_k} ON cases from top to bottom before concluding.
"""
        red_box_notice = ""
        task_mapping_instruction = "by mapping the DARK RED spots"
        task_reasoning_extra = "- What exact physical state or object the DARK RED heatmap is highlighting.\n   - How you formulated the label to act as the COMMON DENOMINATOR across all rows and the intended Action."
    # ----------------------------------------

    prompt = f"""You are an expert AI researcher analyzing Concept Neurons in a Deep Reinforcement Learning agent.

=========================================
**ENVIRONMENT CONTEXT:**
{env_context}
=========================================

I am providing an image containing {top_k} ACTIVATED (ON) cases and {top_k} INACTIVE (OFF) cases for a single concept neuron (total {2 * top_k} scenarios to analyze). 
The image uses a CONTRASTIVE layout. Each row directly compares an ACTIVATED state (ON) vs an INACTIVE state (OFF) for this concept. 
There are 6 panels from left to right in each row:

--- ACTIVATED EXAMPLES (Panels 1-3) ---
1. Panel 1 (labeled [God View (ON)] or [Full Render (ON)]): The global, uncropped view of the game environment.
2. Panel 2 (labeled [Agent View (ON)] or [Model Input (ON)]): The agent's actual visual observation. {red_box_notice}
3. Panel 3 (labeled [IG Heatmap (ON)]): The Integrated Gradients attribution map using a 'JET' colormap. This map aligns perfectly 1:1 with Panel 2.
   *** HEATMAP COLOR SCALE ***
   - DARK RED / RED: Local attention / Highest attribution.
   - YELLOW / GREEN: Medium to low attribution.
   - DARK BLUE: Zero attribution (background/ignored).

--- INACTIVE EXAMPLES (Panels 4-6) ---
4. Panel 4 (labeled [God View (OFF)] or [Full Render (OFF)]): The global view when the concept is NOT activated.
5. Panel 5 (labeled [Agent View (OFF)] or [Model Input (OFF)]): The agent's observation when the concept is OFF.
6. Panel 6 (labeled [IG Heatmap (OFF)]): The attribution map when deactivated.
"""

    if rules_context:
        prompt += f"""
=========================================
**LOGICAL CONTEXT (CRITICAL HINT):**
The AI agent explicitly uses this concept to make decisions. Here are the deduplicated rules involving this concept:
{rules_context}
=========================================
"""

    prompt += f"""
=========================================
**CRITICAL VISUAL ANALYSIS RULES (DO NOT IGNORE):**
{visual_rules}
=========================================

Your task:
1. Perform Contrastive Analysis: Compare ON vs OFF panels across ALL {top_k} rows. Anchor your analysis on the DARK RED regions to find the focal point, but INTERPRET that focal point within the holistic context of the God View / Full Render and the position {task_mapping_instruction}. Avoid tunnel vision!
2. Evaluate Monosemanticity (Consistency vs Exceptions): Look closely at EVERY SINGLE ON row. Does the concept consistently represent one functional state or holistic situation?
3. Assign a 'monoscore' from 0 to 5:
   - 5: Perfectly monosemantic (100% consistent functionally or visually across ALL ON panels).
   - 3-4: Mostly monosemantic, but with 1 or 2 clear exceptions/deviations.
   - 1-2: Polysemantic (triggers on multiple distinct, functionally unrelated features).
   - 0: Completely uninterpretable or random.
4. Generate a concise, descriptive snake_case label. **CRITICAL:** The label MUST be the ultimate common denominator that logically explains ALL {top_k} ON cases AND perfectly justifies the intended Action. Use prefixes like `likely_` if the visual evidence is mostly contextual (e.g., "Zero Attrib" rows) but the global view and Rules strongly imply a specific state.
5. Assign a confidence score from 1 to 5 for your overall assessment.
6. Provide reasoning explaining the contrast. You MUST explicitly state:
   - What the God View / Full Render reveals about the overall situation.
   {task_reasoning_extra}

Output EXACTLY in this JSON format:
{{
  "concept_id": "C_ID_FROM_TITLE",
  "label": "your_snake_case_label",
  "monoscore": 4,
  "confidence": 4,
  "reasoning": "Explain the contrast, emphasizing the holistic synthesis to find the COMMON DENOMINATOR across all rows, spatial mapping, and how the label justifies the Action."
}}
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
    
    preview_count = min(10, len(results["concepts"]))
    for i, (cid, data) in enumerate(results["concepts"].items()):
        if i >= preview_count: break
        print(f"  - {cid}: {data['label']} (Mono: {data['monoscore']}/5, Conf: {data['confidence']}/5)")

if __name__ == "__main__":
    main()