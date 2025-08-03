import os
import time
import argparse
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import torch.nn.functional as F
import random
import re
import difflib

parser = argparse.ArgumentParser(description='TinyStories text continuation generator')
parser.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Directory containing model checkpoint')
parser.add_argument('--prompt', type=str, required=True,
                    help='Initial text to continue from')
parser.add_argument('--num_samples', type=int, default=1,
                    help='Number of different continuations to generate')
parser.add_argument('--max_new_tokens', type=int, default=250,
                    help='Maximum new tokens to generate')
parser.add_argument('--temperature', type=float, default=0.6,
                    help='Sampling temperature (0.5-0.7 recommended)')
parser.add_argument('--top_k', type=int, default=40,
                    help='Top-k filtering threshold (0=disabled)')
parser.add_argument('--top_p', type=float, default=0.9,
                    help='Nucleus sampling threshold')
parser.add_argument('--repetition_penalty', type=float, default=1.1,
                    help='Penalty for repeated tokens (1.0=none, 1.1-1.3 recommended)')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed for reproducibility (default: random)')
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'], help='Compute device')
parser.add_argument('--coherence_mode', type=str, default='enhanced',
                    choices=['basic', 'enhanced', 'advanced'],
                    help='Level of coherence enforcement')
parser.add_argument('--dialogue_limit', type=int, default=10,
                    help='Maximum back-and-forth dialogue turns before forcing progression')
parser.add_argument('--consistency_check_frequency', type=int, default=50,
                    help='How often to check for logical consistency (in tokens)')
parser.add_argument('--fix_incomplete_sentences', action='store_true',
                    help='Ensure output ends with complete sentences')
args = parser.parse_args()

# Setup
if args.seed is None:
    args.seed = random.randint(0, 2**32-1)
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

# Device setup
device = args.device
ctx = nullcontext()  # No special context for CPU

# Load model
print(f"Loading model from {args.checkpoint_dir}...")
ckpt_path = os.path.join(args.checkpoint_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to(device)
print(f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# -----------------------------------------------------------------------------
# Tokenizer setup
enc = tiktoken.get_encoding('gpt2')

# -----------------------------------------------------------------------------
# Story markers for detecting story boundaries
STORY_BEGINNINGS = [
    "Once upon a time",
    "One day",
    "There was a",
    "There once was",
    "Long ago",
    "Once there was",
    "In a faraway"
]

STORY_ENDINGS = [
    "The end.",
    "THE END",
    "the end.",
    "And they lived happily ever after.",
    "That was the end of",
    "From that day on,"
]

# Story progression markers
MIDDLE_MARKERS = ["Later", "After that", "Soon", "Next", "Then"]
ENDING_MARKERS = ["Finally", "In the end", "At last", "The next day"]

# -----------------------------------------------------------------------------
# Coherence improvement helpers

def extract_character_names(text):
    """Extract potential character names from text"""
    # Simple pattern for proper nouns in English text
    name_pattern = r'\b[A-Z][a-z]+\b'
    potential_names = re.findall(name_pattern, text)

    # Filter out common non-character capitalized words
    common_words = {'The', 'A', 'An', 'It', 'They', 'We', 'You', 'He', 'She', 'His', 'Her', 'I', 'My', 'Your', 'Their',
                   'Then', 'There', 'This', 'That', 'When', 'Where', 'Why', 'How', 'What', 'Who', 'But', 'And', 'Or', 'If'}
    names = [name for name in potential_names if name not in common_words]

    return set(names)

def extract_locations(text):
    """Extract potential locations from text"""
    # Simple patterns like "in the X", "at the X"
    location_patterns = [
        r'in the ([A-Za-z]+ ?[A-Za-z]*)',
        r'at the ([A-Za-z]+ ?[A-Za-z]*)',
        r'to the ([A-Za-z]+ ?[A-Za-z]*)'
    ]

    locations = set()
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match.lower() not in ['door', 'time', 'morning', 'afternoon', 'night', 'way', 'thing']:
                locations.add(match)

    return locations

def extract_possessions(text, character_names):
    """Extract possessions associated with characters"""
    possessions = {}

    for name in character_names:
        # Look for patterns like "Name's X" or "Name had a X"
        patterns = [
            fr"{name}'s ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} had a ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} had an ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} owned a ([A-Za-z]+ ?[A-Za-z]*)"
        ]

        name_possessions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            name_possessions.update(matches)

        if name_possessions:
            possessions[name] = name_possessions

    return possessions

def extract_attitudes(text, character_names):
    """Extract character attitudes and opinions"""
    attitudes = {}

    for name in character_names:
        # Look for patterns like "Name wanted to X" or "Name didn't like X"
        patterns = [
            fr"{name} wanted to ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} didn't want to ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} liked ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} didn't like ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} loved ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} hated ([A-Za-z]+ ?[A-Za-z]*)",
            fr"{name} was ([A-Za-z]+ ?[A-Za-z]*)"  # Character states
        ]

        char_attitudes = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Store the full attitude statement
                attitude_statement = re.search(fr"{name}[^\.]+{match}[^\.]", text)
                if attitude_statement:
                    char_attitudes.append(attitude_statement.group(0))
                else:
                    char_attitudes.append(f"{name} {match}")

        if char_attitudes:
            attitudes[name] = char_attitudes

    return attitudes

def extract_story_context(text):
    """Extract key story context information"""
    context = {
        'characters': extract_character_names(text),
        'locations': extract_locations(text),
        'possessions': {},
        'attitudes': {},
        'activities': set(),
    }

    # Extract possessions after we have characters
    context['possessions'] = extract_possessions(text, context['characters'])

    # Extract attitudes and opinions
    context['attitudes'] = extract_attitudes(text, context['characters'])

    # Simple activity detection (verbs following character names)
    for character in context['characters']:
        activity_pattern = fr'{character} ([a-z]+ed|[a-z]+ing)'
        activities = re.findall(activity_pattern, text)
        if activities:
            context['activities'].update(activities)

    return context

def extract_dialogue(text):
    """Extract dialogue from the text"""
    dialogue_pattern = r'"([^"]+)"'
    return re.findall(dialogue_pattern, text)

def check_dialogue_consistency(dialogues):
    """Check for consistency in dialogue exchanges"""
    inconsistencies = []

    if len(dialogues) < 4:  # Need at least 2 exchanges to check
        return inconsistencies

    # Check for direct contradictions in responses
    for i in range(len(dialogues) - 1):
        # Check for yes/no conflicts
        if ("Yes" in dialogues[i] or "yes" in dialogues[i]) and \
           ("No" in dialogues[i+1] or "no" in dialogues[i+1]) and \
           similar_content(dialogues[i], dialogues[i+1]):
            inconsistencies.append("Yes/No contradiction in dialogue")

        # Check for can/can't conflicts
        if ("can " in dialogues[i].lower() and "can't " in dialogues[i+1].lower() or
            "can't " in dialogues[i].lower() and "can " in dialogues[i+1].lower()) and \
           similar_content(dialogues[i], dialogues[i+1]):
            inconsistencies.append("Can/Can't contradiction in dialogue")

        # Check for will/won't conflicts
        if ("will " in dialogues[i].lower() and "won't " in dialogues[i+1].lower() or
            "won't " in dialogues[i].lower() and "will " in dialogues[i+1].lower()) and \
           similar_content(dialogues[i], dialogues[i+1]):
            inconsistencies.append("Will/Won't contradiction in dialogue")

    return inconsistencies

def similar_content(text1, text2):
    """Check if two pieces of text are talking about the same thing"""
    # Simple check if they share significant words
    words1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text2.lower()))

    # If they share at least 3 significant words
    return len(words1.intersection(words2)) >= 3

def check_logical_coherence(new_text, context):
    """Enhanced check for logical inconsistencies in new text given context"""
    # Extract context from new text
    new_context = extract_story_context(new_text)
    inconsistencies = []

    # Check for new characters appearing suddenly late in the story
    if len(context['characters']) >= 2:  # We have established characters
        for new_char in new_context['characters']:
            if new_char not in context['characters']:
                inconsistencies.append(f"New character '{new_char}' appears without introduction")

    # Check for sudden location changes without transition
    if context['locations'] and new_context['locations']:
        current_loc = list(context['locations'])[-1] if context['locations'] else None
        new_locs = list(new_context['locations'])

        if new_locs and current_loc != new_locs[0]:
            # Look for transition words
            transition_words = ['went', 'walked', 'moved', 'traveled', 'left', 'arrived']
            has_transition = any(word in new_text.lower() for word in transition_words)

            if not has_transition:
                inconsistencies.append(f"Location changed from '{current_loc}' to '{new_locs[0]}' without transition")

    # Check possession consistency
    for character, items in context['possessions'].items():
        for item in items:
            # Check for contradictions like "X had no Y" when we know "X had a Y"
            if f"{character} had no {item}" in new_text or f"{character} didn't have a {item}" in new_text:
                inconsistencies.append(f"Contradiction: {character} previously had {item} but now doesn't")

    # Check for attitude consistency
    for character, attitudes in context['attitudes'].items():
        # Check for contradictions in character attitudes
        if character in new_context['attitudes']:
            new_attitudes = new_context['attitudes'][character]

            for old_att in attitudes:
                for new_att in new_attitudes:
                    # Check for contradictions like "liked" vs "didn't like"
                    if ("liked" in old_att and "didn't like" in new_att) or \
                       ("didn't like" in old_att and "liked" in new_att):
                        inconsistencies.append(f"Contradiction in {character}'s attitudes")

                    # Check for "wanted to" vs "didn't want to"
                    if ("wanted to" in old_att and "didn't want to" in new_att) or \
                       ("didn't want to" in old_att and "wanted to" in new_att):
                        inconsistencies.append(f"Contradiction in {character}'s desires")

    # Extract dialogue for consistency checks
    dialogues = extract_dialogue(context.get('previous_text', '') + new_text)
    dialogue_inconsistencies = check_dialogue_consistency(dialogues)
    inconsistencies.extend(dialogue_inconsistencies)

    # Check for direct contradictions in statements
    contradictions = [
        (r"can't ([a-z]+)", r"can \1"),  # 检测"can't X" vs "can X"
        (r"don't ([a-z]+)", r"do \1"),   # 检测"don't X" vs "do X"
        (r"won't ([a-z]+)", r"will \1"), # 检测"won't X" vs "will X"
        (r"isn't ([a-z]+)", r"is \1"),   # 检测"isn't X" vs "is X"
    ]

    full_text = " ".join([context.get('previous_text', ''), new_text])

    for pattern_neg, pattern_pos in contradictions:

        neg_statements = re.findall(pattern_neg, full_text, re.IGNORECASE)

        for statement in neg_statements:
            pos_pattern = pattern_pos.replace(r"\1", re.escape(statement))

            if re.search(pos_pattern, full_text, re.IGNORECASE):
                inconsistencies.append(f"Contradiction: Both can and cannot '{statement}'")

    return inconsistencies

def is_repetitive_dialogue(dialogues, threshold=0.7):
    """Check if dialogue is becoming repetitive"""
    if len(dialogues) < 6:  # Need enough dialogue to detect patterns
        return False

    # Check last 3 exchanges (6 dialogue pieces)
    recent_dialogues = dialogues[-6:]

    # Check for repeated content
    for i in range(len(recent_dialogues) - 2):
        similarity = difflib.SequenceMatcher(None, recent_dialogues[i], recent_dialogues[i+2]).ratio()
        if similarity > threshold:
            return True

    return False

def dynamic_temperature_adjustment(temp_base, inconsistencies, progress, repetitive_dialogue=False):
    """Dynamically adjust temperature based on inconsistencies and story progress"""
    # Start with base temperature
    temp = temp_base

    # Reduce temperature if we have inconsistencies (makes model more conservative)
    if inconsistencies:
        temp -= min(0.2, 0.05 * len(inconsistencies))

    # Reduce temperature even more for repetitive dialogue
    if repetitive_dialogue:
        temp -= 0.15

    # Generally decrease temperature as we approach the end (for coherent ending)
    if progress > 0.7:
        temp -= 0.1

    # Ensure we stay in reasonable range
    return max(0.4, min(0.9, temp))

def ensure_complete_sentence(text):
    """Ensure text ends with a complete sentence"""
    # Check if it already ends with sentence-ending punctuation
    if re.search(r'[.!?]$', text):
        return text

    # Find the last sentence-ending punctuation
    last_period = text.rfind('.')
    last_exclam = text.rfind('!')
    last_question = text.rfind('?')

    # Get the position of the last sentence end
    last_end = max(last_period, last_exclam, last_question)

    # If we found a sentence end, truncate to that point
    if last_end > 0:
        # Make sure we're not breaking inside a quotation
        quote_after = text.find('"', last_end)
        if quote_after > 0:  # There's a quote after the last period
            # Check if there's an opening quote before
            last_quote_before = text.rfind('"', 0, last_end)
            if last_quote_before >= 0 and text.count('"', last_quote_before, quote_after) % 2 == 1:
                # We're inside a quote, find the closing quote
                return text[:quote_after+1]

        return text[:last_end+1]

    return text  # Can't find a good spot to end, return as is

# -----------------------------------------------------------------------------
# Enhanced text generation function
def generate_text(prompt, max_tokens=250, temp=0.6, top_k=40, top_p=0.9, rep_penalty=1.1,
                  coherence_mode='enhanced', consistency_check_frequency=50, dialogue_limit=10,
                  fix_incomplete_sentences=False):
    """Generate text continuation with logical coherence enhancements"""
    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long)[None, ...].to(device)

    # Track generated tokens and text
    generated_ids = []
    full_text = prompt

    # Extract initial context from prompt
    story_context = extract_story_context(prompt)
    story_context['previous_text'] = prompt
    context_update_frequency = consistency_check_frequency

    # For tracking story state
    min_length_reached = False
    inconsistencies = []
    generation_attempts = 0
    temp_base = temp  # Save original temperature for adjustments

    # For dialogue tracking
    dialogue_turns = 0
    all_dialogues = extract_dialogue(prompt)
    repetitive_dialogue = False

    # Story phase tracking
    story_phase = "beginning" if len(prompt) < 100 else "middle"

    with torch.no_grad():
        for i in range(max_tokens):
            # Progress through the story (0 to 1)
            progress = i / max_tokens

            # Dynamic temperature adjustment based on coherence needs
            if coherence_mode in ['enhanced', 'advanced']:
                # Adjust temperature based on inconsistencies, progress, and dialogue state
                temp = dynamic_temperature_adjustment(temp_base, inconsistencies, progress, repetitive_dialogue)

            # Forward pass through the model
            logits, _ = model(x)
            logits = logits[:, -1, :] / temp

            # Apply repetition penalty
            if rep_penalty > 1.0:
                for token_id in set(x[0].tolist()[-20:]):  # Only penalize recent tokens
                    logits[0, token_id] /= rep_penalty

            # Advanced coherence mode: apply extra penalties to avoid inconsistencies
            if coherence_mode == 'advanced' and len(inconsistencies) > 0:
                # If we're in advanced mode and detect inconsistencies,
                # we can try to nudge the model away from those patterns

                # This is a simplified approach - in a more complex implementation,
                # we would identify specific tokens that could lead to contradictions

                # For now, we'll just increase the repetition penalty to make
                # the model less likely to repeat problematic patterns
                for token_id in set(x[0].tolist()[-50:]):  # Look at a larger window
                    logits[0, token_id] /= (rep_penalty * 1.2)  # Increased penalty

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[:, [-1]]] = -float('Inf')

            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add to generated sequence
            generated_ids.append(next_token.item())
            x = torch.cat((x, next_token), dim=1)

            # Every few tokens, check the coherence
            if (i + 1) % context_update_frequency == 0 and i > 0:
                # Decode recent generation
                recent_text = enc.decode(generated_ids[-context_update_frequency:])
                full_text += recent_text

                # Track dialogues
                new_dialogues = extract_dialogue(recent_text)
                if new_dialogues:
                    all_dialogues.extend(new_dialogues)
                    dialogue_turns = len(all_dialogues) // 2  # Estimate turns as pairs of dialogue

                    # Check for repetitive dialogue
                    repetitive_dialogue = is_repetitive_dialogue(all_dialogues)

                    if dialogue_turns > dialogue_limit or repetitive_dialogue:
                        # Try to push the story forward by injecting a transition event
                        if coherence_mode == 'advanced':
                            # In a more complex implementation, we would influence
                            # the next generation to break the dialogue pattern
                            pass

                # Detect story phase transitions
                recent_lower = recent_text.lower()
                if story_phase == "beginning" and any(marker.lower() in recent_lower for marker in MIDDLE_MARKERS):
                    story_phase = "middle"
                elif story_phase == "middle" and any(marker.lower() in recent_lower for marker in ENDING_MARKERS):
                    story_phase = "ending"

                # Enhanced and Advanced coherence modes check for logical issues
                if coherence_mode in ['enhanced', 'advanced']:
                    # Check for inconsistencies
                    inconsistencies = check_logical_coherence(recent_text, story_context)

                    # Advanced mode: Consider regenerating problematic sections
                    if coherence_mode == 'advanced' and inconsistencies and generation_attempts < 3:
                        # If we find inconsistencies, we take more aggressive action
                        generation_attempts += 1
                        temp *= 0.8  # Reduce temperature to make generation more conservative

                    # Update our story context
                    new_context = extract_story_context(recent_text)
                    story_context['characters'].update(new_context['characters'])
                    story_context['locations'].update(new_context['locations'])

                    # Update possessions
                    for char, items in new_context['possessions'].items():
                        if char not in story_context['possessions']:
                            story_context['possessions'][char] = items
                        else:
                            story_context['possessions'][char].update(items)

                    # Update attitudes
                    for char, attitudes in new_context['attitudes'].items():
                        if char not in story_context['attitudes']:
                            story_context['attitudes'][char] = attitudes
                        else:
                            story_context['attitudes'][char].extend(attitudes)

                    story_context['activities'].update(new_context['activities'])
                    story_context['previous_text'] = full_text

            # Check if minimum length is reached
            if len(generated_ids) >= 100:
                min_length_reached = True

            # Get the recent text for checking stopping conditions
            recent_text = enc.decode(generated_ids[-50:])

            # Check for story completion conditions
            if min_length_reached:
                # Check for a new story beginning (which means our story has likely ended)
                for beginning in STORY_BEGINNINGS:
                    if beginning in recent_text and len(generated_ids) > 120:
                        # Find where the new story begins and trim
                        cut_index = recent_text.find(beginning)
                        if cut_index > 0:
                            # Cut the generated IDs at appropriate point
                            trim_count = len(enc.encode(recent_text[cut_index:]))
                            generated_ids = generated_ids[:-trim_count]
                            text = enc.decode(generated_ids).replace("<|PARA|>", "\n\n")
                            return ensure_complete_sentence(text) if fix_incomplete_sentences else text

                # Check for story endings
                for ending in STORY_ENDINGS:
                    if ending in recent_text:
                        # Find where in the recent text the ending occurs
                        end_pos = recent_text.find(ending) + len(ending)
                        if end_pos < len(recent_text):
                            # Trim any content after the ending
                            trim_count = len(enc.encode(recent_text[end_pos:]))
                            if trim_count < len(generated_ids):
                                generated_ids = generated_ids[:-trim_count]
                        text = enc.decode(generated_ids).replace("<|PARA|>", "\n\n")
                        return ensure_complete_sentence(text) if fix_incomplete_sentences else text

                # If we have a long-enough story and see a sentence end, consider stopping
                if len(generated_ids) > 200 and any(recent_text.endswith(end) for end in ['.', '!', '?']):
                    # Make sure it's not ending mid-sentence like "Mr." or "No."
                    if not any(recent_text.endswith(abbr) for abbr in ["Mr.", "Mrs.", "Dr.", "No.", "etc."]):
                        if len(generated_ids) > max_tokens * 0.8:  # Over 80% of max length
                            break

            # Check if we've reached the max length
            if len(generated_ids) >= max_tokens - 10:
                # Try to end at a sentence boundary
                if any(recent_text.endswith(end) for end in ['.', '!', '?']):
                    break

    # Process the final text
    generated_text = enc.decode(generated_ids).replace("<|PARA|>", "\n\n")

    # Ensure complete sentences if requested
    if fix_incomplete_sentences:
        generated_text = ensure_complete_sentence(generated_text)

    return generated_text

# -----------------------------------------------------------------------------
# Main execution
print(f"Seed: {args.seed}")
print(f"Prompt: {args.prompt}")
print(f"Coherence mode: {args.coherence_mode}")
print('-' * 60)

for i in range(args.num_samples):
    # Set seed for reproducibility within the sample
    sample_seed = args.seed + i
    torch.manual_seed(sample_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(sample_seed)

    # Generate text continuation
    continuation = generate_text(
        args.prompt,
        max_tokens=args.max_new_tokens,
        temp=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.repetition_penalty,
        coherence_mode=args.coherence_mode,
        consistency_check_frequency=args.consistency_check_frequency,
        dialogue_limit=args.dialogue_limit,
        fix_incomplete_sentences=args.fix_incomplete_sentences
    )

    # Print the result
    print(f"Sample {i+1} (seed: {sample_seed}):")
    full_text = args.prompt + continuation
    print(full_text)
    print('-' * 60)

# Print parameters used
print(f"Generation complete. Used parameters: coherence_mode={args.coherence_mode}, temp={args.temperature}, " +
      f"top_k={args.top_k}, top_p={args.top_p}, rep_penalty={args.repetition_penalty}")