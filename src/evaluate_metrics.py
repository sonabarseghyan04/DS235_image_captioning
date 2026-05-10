import os
import json
import csv
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def evaluate_model(ground_truth_path, results_path):
    print(f"\n" + "=" * 50)
    print(f"Evaluating: {results_path}")
    print("=" * 50)

    coco = COCO(ground_truth_path)
    coco_res = coco.loadRes(results_path)

    #Creating the evaluator object
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    print("Calculating metrics...")
    coco_eval.evaluate()

    print("\n Final Scores:")
    for metric, score in coco_eval.eval.items():
        print(f"   {metric:>10}: {score:.4f}")

    return coco_eval.eval


if __name__ == "__main__":
    ground_truth_file = "results/ground_truth.json"

    if not os.path.exists(ground_truth_file):
        print(f"Error: Cannot find {ground_truth_file}. Run data_cleaner.py first!")
        exit()

    model_results = {
        "BLIP Base": "results/blip_full_results.json",
        "BLIP-2 (OPT-2.7b)": "results/blip2_full_results.json",
        "GIT Base": "results/git_full_results.json"
    }

    all_metrics = {}

    for model_name, result_file in model_results.items():
        if os.path.exists(result_file):
            scores = evaluate_model(ground_truth_file, result_file)
            all_metrics[model_name] = scores
        else:
            print(f"\n Skipping {model_name} - File not found.")
            print(f"   (Missing: {result_file})")

    if all_metrics:
        output_csv = "results/quantitative_scores.csv"
        output_json = "results/quantitative_scores.json"


        headers = ["Model", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for model_name, scores in all_metrics.items():
                row = [model_name]
                for metric in headers[1:]:

                    row.append(f"{scores.get(metric, 0.0):.4f}")
                writer.writerow(row)

        with open(output_json, "w") as f:
            json.dump(all_metrics, f, indent=4)

        print("\n Evaluation complete!")
        print(f" {output_csv} (Spreadsheet format)")
        print(f" {output_json} (Raw data)")