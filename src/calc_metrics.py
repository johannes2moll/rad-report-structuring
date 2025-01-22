import json
from datasets import load_dataset

from utils import get_lists, parse_args_calc
from StructEval.structeval.StructEval import StructEval

def eval_findings(ref_findings_list, pred_findings_list):
    """Evaluate the findings section of the radiology reports."""
    assert len(ref_findings_list) == len(pred_findings_list)
    # Check pred_list for None values
    if None in pred_findings_list:
        # find out which index has None value
        none_indices = [i for i, x in enumerate(pred_findings_list) if x == None]
        print("Warning: None values found in predictions.")
        print("None values found at indices:", none_indices)
        # remove from both lists
        ref_findings_list = [i for j, i in enumerate(ref_findings_list) if j not in none_indices]
        pred_findings_list = [i for j, i in enumerate(pred_findings_list) if j not in none_indices]
        print("Removed corresponding entry from reference and prediction lists.")
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)
    
    results = rr.forward(refs=ref_findings_list, hyps=pred_findings_list, section="findings", aligned=False)
    return results

def eval_impression(ref_impression_list, pred_impression_list):
    """Evaluate the impression section of the radiology reports."""
    assert len(ref_impression_list) == len(pred_impression_list)
    # Check pred_list for None values
    if None in pred_impression_list:
        # find out which index has None value
        none_indices = [i for i, x in enumerate(pred_impression_list) if x == None]
        print("Warning: None values found in predictions.")
        print("None values found at indices:", none_indices)
        # remove from both lists
        ref_impression_list = [i for j, i in enumerate(ref_impression_list) if j not in none_indices]
        pred_impression_list = [i for j, i in enumerate(pred_impression_list) if j not in none_indices]
        print("Removed corresponding entry from reference and prediction lists.")
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)
    
    results = rr.forward(refs=ref_impression_list, hyps=pred_impression_list, section="impression", aligned=False)
    return results

def main():
    args = parse_args_calc()
    
    ref_findings_mimic_list, ref_impression_mimic_list, pred_findings_mimic_list, pred_impression_mimic_list = get_lists(args.pred_file_mimic, args.ref_data_path, dataset_name="mimic", split="test_reviewed")
    ref_findings_chexpert_list, ref_impression_chexpert_list, pred_findings_chexpert_list, pred_impression_chexpert_list = get_lists(args.pred_file_chexpert, args.ref_data_path, dataset_name="chexpert", split="test_reviewed")
    try:
        print("Calculating metrics for the MIMIC findings section...")
        findings_mimic_results = eval_findings(ref_findings_mimic_list, pred_findings_mimic_list)
        print("Calculating metrics for the MIMIC impression section...")
        impression_mimic_results = eval_impression(ref_impression_mimic_list, pred_impression_mimic_list)
    except: 
        findings_mimic_results = None
        impression_mimic_results = None
    try:
        print("Calculating metrics for the CheXpert findings section...")
        findings_chexpert_results = eval_findings(ref_findings_chexpert_list, pred_findings_chexpert_list) 
        print("Calculating metrics for the CheXpert impression section...")
        impression_chexpert_results = eval_impression(ref_impression_chexpert_list, pred_impression_chexpert_list)
    except: 
        findings_chexpert_results = None
        impression_chexpert_results = None
    # write to file
    with open(args.output_file, "w") as f:
        json.dump({"findings MIMIC": findings_mimic_results, "impression MIMIC": impression_mimic_results}, f)
        json.dump({"findings CheXpert": findings_chexpert_results, "impression CheXpert": impression_chexpert_results}, f)

    print("Results saved to file:", args.output_file)
    # Print results
    print("Findings MIMIC Results:")
    print(findings_mimic_results)
    print("\nImpression MIMIC Results:")
    print(impression_mimic_results)
    print("\nFindings CheXpert Results:")
    print(findings_chexpert_results)
    print("\nImpression CheXpert Results:")
    print(impression_chexpert_results)

if __name__ == "__main__":
    main()
