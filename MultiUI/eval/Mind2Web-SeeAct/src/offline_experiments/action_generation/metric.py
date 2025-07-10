import json
import os
from collections import defaultdict

predict_folder = 'Mind2Web-SeeAct/offline_output_bbox_gt_crop_gen/bbox_generate_gt_crop_offline_data_-1choices/qwen2_mind2web/test_website/'
gt_folder = '/netscratch/banwari/llm_energy/MultiUI/screenshot_generation/data/Mind2Web_bbox_eval/bbox_generate_gt_crop_offline_data_-1choices/test_website/'

def check_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    if x2_1 < x1_2 or x2_2 < x1_1: return False
    if y2_1 < y1_2 or y2_2 < y1_1: return False
    return True

def check_center_point(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_c = (x1_1 + x2_1) / 2
    y_c = (y1_1 + y2_1) / 2
    return x1_2 <= x_c <= x2_2 and y1_2 <= y_c <= y2_2

element_acc = defaultdict(int)
step_sr = defaultdict(int)
err_dict = defaultdict(int)

for fn in os.listdir(predict_folder):
    print(f"\nProcessing prediction file: {fn}")
    try:
        pred_path = os.path.join(predict_folder, fn)
        entry = json.loads(open(pred_path, 'r').readlines()[0])
        raw_output = entry['gpt_output'][0]

        print(f"Raw prediction: {raw_output}")

        predicted_coord = raw_output.split('[')[-1].split(']')[0]
        predicted_coord = [float(num) for num in predicted_coord.split(',')]
        print(f"Predicted bbox: {predicted_coord}")

        try:
            predicted_action = raw_output.split('And my action is Action: ')[-1].split('\n')[0].strip()
        except:
            predicted_action = ''
        print(f"Predicted action: {predicted_action}")

        if 'Value: ' in raw_output:
            predicted_value = raw_output.split('Value: ')[-1].strip()
        else:
            predicted_value = ''
        print(f"Predicted value: {predicted_value}")

        annotation_id = fn.split('_predictions_bbox')[0]

        # Load GT bounding box
        gt_file_path = os.path.join(gt_folder, annotation_id, 'queries.jsonl')
        gt_entry = json.loads(open(gt_file_path, 'r').readlines()[0])
        gt_coord = gt_entry['bbox_ratio_xyxy']
        print(f"GT bbox: {gt_coord}")

        task_type = gt_folder.strip('/').split("/")[-1]
        gt_lines_path = os.path.join(f'/netscratch/banwari/llm_energy/MultiUI/screenshot_generation/data/Mind2Web_bbox_eval/offline_data_-1choices/{task_type}/', annotation_id, 'queries.jsonl')
        lines = open(gt_lines_path, 'r').readlines()
        print(f"Loaded {len(lines)} lines from offline GT")

        for line in lines:
            gt_entry = json.loads(line)
            if 'target' not in gt_entry: continue
            actn = '\n'.join(gt_entry['target'].split('\n')[1:]).strip()
            if len(actn) > 1:
                gt_action = gt_entry['target'].split('\n')[1].replace('Action: ', '').strip()
                try:
                    gt_value = gt_entry['target'].split('\n')[2].replace('Value: ', '').strip()
                except:
                    gt_value = ''
                break
        print(f"GT action: {gt_action} | GT value: {gt_value}")

        match_box = check_center_point(predicted_coord, gt_coord)
        match_action = predicted_action == gt_action
        match_value = predicted_value == gt_value

        print(f"Match center: {match_box}, Match action: {match_action}, Match value: {match_value}")

        element_acc[match_box] += 1
        step_sr[match_box and match_action and match_value] += 1

    except Exception as e:
        print(f"Error processing {fn}: {e}")
        err_dict['fault'] += 1

# Final Stats
print("\n\nEvaluation Summary:")
print("Errors:", err_dict)
if element_acc:
    print('Element Accuracy:', element_acc, f"{100 * element_acc[True] / sum(element_acc.values()):.2f}%")
if step_sr:
    print('Step Success Rate:', step_sr, f"{100 * step_sr[True] / sum(step_sr.values()):.2f}%")
