import json
from utils import get_lists, parse_args_calc
from StructEval.structeval.StructEval import StructEval

def eval_findings(ref_findings_list, pred_findings_list):
    """Evaluate the findings section of the radiology reports."""
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

    ref_findings_list, ref_impression_list, pred_findings_list, pred_impression_list = get_lists(args.pred_file, args.ref_file)
    
    print("Calculating metrics for the findings section...")
    findings_results = eval_findings(ref_findings_list, pred_findings_list)
    print("Calculating metrics for the impression section...")
    impression_results = eval_impression(ref_impression_list, pred_impression_list)
    
    # write to file
    with open(args.output_file, "w") as f:
        json.dump({"findings": findings_results, "impression": impression_results}, f)
    print("Results saved to file:", args.output_file)
    # Print results
    print("Findings Results:")
    print(findings_results)
    print("\nImpression Results:")
    print(impression_results)

if __name__ == "__main__":
    main()
