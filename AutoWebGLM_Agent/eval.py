import json
import sys
import re
import numpy as np

from rouge_chinese import Rouge
import jieba

#################################### Import carbontracker and apply the fix ---- ONLY NEEDED IF RUNNING ON SLURM CLUSTER
try:
    from carbontracker.tracker import CarbonTracker, CarbonTrackerThread
    
    # Save original method
    original_log_components_info = CarbonTrackerThread._log_components_info
    
    # Create fixed method
    def fixed_log_components_info(self):
        log = ["The following components were found:"]
        for comp in self.components:
            name = comp.name.upper()
            # Fix here: decode byte strings in device names
            devices = [d.decode('utf-8') if isinstance(d, bytes) else d for d in comp.devices()]
            devices = ", ".join(devices)
            log.append(f"{name} with device(s) {devices}.")
        log_str = " ".join(log)
        print(log_str)
    
    # Apply the patch
    CarbonTrackerThread._log_components_info = fixed_log_components_info
    print("Successfully patched carbontracker device handling")
    
except (ImportError, AttributeError) as e:
    print(f"Failed to set up carbontracker: {e}")
##########################################################################################

def get_rouge_score(hypothesis, reference):
    if hypothesis is None or reference is None:
        return None

    hypothesis = ' '.join(jieba.cut(hypothesis)) 
    reference = ' '.join(jieba.cut(reference))

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)

    return scores[0]["rouge-1"]['f']

def extract_line(text):
    """Extract type, label, and param from one line like:
    [textbox] Destination -> TYPE: Pune
    """
    ans = {'type': None, 'label': None, 'param': None}
    match = re.match(r"\[(.*?)\]\s*(.*?)\s*->\s*(\w+)(?::\s*(.*))?", text.strip())
    if match:
        ans['label'] = match.group(2).strip()
        ans['type'] = match.group(3).strip().upper()
        if match.group(4):
            ans['param'] = match.group(4).strip()
    return ans

def extract_all_actions(multiline_text):
    """Split the text by lines and extract each step."""
    actions = []
    for line in multiline_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        actions.append(extract_line(line))
    return actions

if __name__ == '__main__':

    # Initialize the carbon tracker
    tracker = CarbonTracker(epochs=1)

    print("Starting carbon tracking")
    tracker.epoch_start()

    result_path = sys.argv[1]

    res_list = {
        'type': [],
        'label': [],
        'param': [],
        'all': []
    }

    with open(result_path) as f:
        all_data = json.load(f)

    for ix, r in enumerate(all_data):
        ref_steps = extract_all_actions(r['reference'])
        pred_steps = extract_all_actions(r['prediction'])

        matched = False
        for ref_action in ref_steps:
            for pred_action in pred_steps:
                res = {}
                # Match type
                if ref_action['type'] and pred_action['type']:
                    res['type'] = int(ref_action['type'] == pred_action['type'])
                # Match label
                if ref_action['label'] and pred_action['label']:
                    res['label'] = int(ref_action['label'] == pred_action['label'])
                # Match param (use ROUGE)
                if ref_action['param'] and pred_action['param']:
                    rouge_score = get_rouge_score(ref_action['param'], pred_action['param'])
                    res['param'] = rouge_score if rouge_score is not None else 0.0
                # Match both type and label
                if ref_action['type'] and ref_action['label']:
                    res['all'] = int(
                        ref_action['type'] == pred_action['type'] and
                        ref_action['label'] == pred_action['label']
                    )

                # Save
                for k, v in res.items():
                    res_list[k].append(v)

                # Mark as matched if exact
                if res.get('all') == 1:
                    matched = True
                    break
            if matched:
                break

    # Compute final averages
    for k, v in res_list.items():
        res_list[k] = float(np.mean(v)) if v else 0.0

    print(res_list)

    # End tracking even if an exception occurs
    print("Ending carbon tracking")
    tracker.epoch_end()
    tracker.stop()
