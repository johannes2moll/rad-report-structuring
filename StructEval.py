from collections import defaultdict
import warnings
from torch import nn
import pandas as pd
import numpy as np
from NLG.rouge.rouge import Rouge
from NLG.bleu.bleu import Bleu
from NLG.bertscore.bertscore import BertScore
from radgraph import F1RadGraph
from structeval.StructBert import StructBert
from structeval.constants import leaves_mapping
from structeval.utils import parse_findings, parse_impression, remove_bullets, remove_numbering
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from f1chexbert import F1CheXbert

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


class StructEval(nn.Module):
    def __init__(self,
                 do_radgraph=True,
                 do_green=True,
                 do_bleu=True,
                 do_rouge=True,
                 do_bertscore=True,
                 do_diseases=True,
                 do_chexbert=False,
                 ):
        super(StructEval, self).__init__()

        self.do_radgraph = do_radgraph
        self.do_green = do_green
        self.do_bleu = do_bleu
        self.do_rouge = do_rouge
        self.do_bertscore = do_bertscore
        self.do_diseases = do_diseases
        self.do_chexbert = do_chexbert

        # Initialize scorers only once
        if self.do_radgraph:
            self.radgraph_scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        if self.do_bleu:
            self.bleu_scorer = Bleu()
        if self.do_bertscore:
            self.bertscore_scorer = BertScore(model_type='distilbert-base-uncased',
                                              num_layers=5)
        if self.do_green:
            # Initialize green scorer here if needed
            pass
        if self.do_rouge:
            self.rouge_scorers = {
                "rouge1": Rouge(rouges=["rouge1"]),
                "rouge2": Rouge(rouges=["rouge2"]),
                "rougeL": Rouge(rouges=["rougeL"])
            }
        if self.do_diseases:
            model = "StanfordAIMI/CXR-BERT-Leaves-Diseases-Only"
            self.diseases_model = StructBert(model_id_or_path=model, mapping=leaves_mapping)

        # Store the metric keys
        self.metric_keys = []
        if self.do_radgraph:
            self.metric_keys.extend(["radgraph_simple", "radgraph_partial", "radgraph_complete"])
        if self.do_bleu:
            self.metric_keys.append("bleu")
        if self.do_green:
            self.metric_keys.append("green")
        if self.do_bertscore:
            self.metric_keys.append("bertscore")
        if self.do_rouge:
            self.metric_keys.extend(self.rouge_scorers.keys())
        if self.do_diseases:
            self.metric_keys.extend(["samples_avg_precision", "samples_avg_recall", "samples_avg_f1-score"])
        if self.do_chexbert:
            self.chexbert_scorer = F1CheXbert()

    def forward(self, refs, hyps, section="impression", aligned=True, do_lower_case=False):
        if section not in ["impression", "findings"]:
            raise ValueError("section must be either impression or findings")

        if section == "impression":
            hyps_parsed = [remove_numbering(parse_impression(hyp, do_lower_case=do_lower_case)) for hyp in hyps]
            refs_parsed = [remove_numbering(parse_impression(ref, do_lower_case=do_lower_case)) for ref in refs]

            # this necessary?
            # refs = [reconstruct_impression(r) for r in refs_parsed]
            # hyps = [reconstruct_impression(r) for r in hyps_parsed]

            return self.run_forward(refs,
                                    hyps,
                                    refs_parsed,
                                    hyps_parsed,
                                    aligned,
                                    )

        else:
            hyps_parsed = [parse_findings(hyp, do_lower_case=do_lower_case) for hyp in hyps]
            refs_parsed = [parse_findings(ref, do_lower_case=do_lower_case) for ref in refs]
            hyp_sections_list = []
            ref_sections_list = []

            organ_headers_scores = defaultdict(list)

            # For findings, we evaluate report per report
            for hyp_parsed, ref_parsed in zip(hyps_parsed, refs_parsed):
                ref_sections = set(ref_parsed.keys())
                hyp_sections = set(hyp_parsed.keys())
                all_sections = ref_sections.union(hyp_sections)
                hyp_sections_list.append(list(hyp_sections))
                ref_sections_list.append(list(ref_sections))
                for section in all_sections:
                    # we penalize both missed section in ref and section in hyps not in ref
                    if section in ref_parsed and section in hyp_parsed:
                        ref_utterances = remove_bullets(ref_parsed[section])
                        hyp_utterances = remove_bullets(hyp_parsed[section])

                        # Evaluate section per section
                        scores = self.run_forward(["\n".join(ref_utterances)],
                                                  ["\n".join(hyp_utterances)],
                                                  [ref_utterances],
                                                  [hyp_utterances],
                                                  aligned,
                                                  )
                    else:
                        # Section mismatch between ref and hyp: assign zero scores
                        scores = {metric: 0 for metric in self.metric_keys}

                    organ_headers_scores[section].append(scores)

            # After processing all reports, compute average scores per organ
            organ_avg_scores = {}
            all_scores = []
            for section, scores_list in organ_headers_scores.items():
                df = pd.DataFrame(scores_list)
                avg_scores = df.mean().to_dict()
                organ_avg_scores[section] = avg_scores
                all_scores.extend(scores_list)

            # Compute overall average scores across all organs
            df_all = pd.DataFrame(all_scores)
            overall_avg_scores = df_all.mean().to_dict()

            # Compute section presence scores
            sections_labels = sorted(list(set(label for sublist in ref_sections_list for label in sublist).union(
                set(label for sublist in hyp_sections_list for label in sublist))))

            def binary_representation(labels, all_labels):
                return [1 if label in labels else 0 for label in all_labels]

            y_true = [binary_representation(ref, sections_labels) for ref in ref_sections_list]
            y_pred = [binary_representation(hyp, sections_labels) for hyp in hyp_sections_list]

            classification_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            section_scores = {}
            section_scores["section_avg_precision"] = classification_dict["samples avg"]["precision"]
            section_scores["section_avg_recall"] = classification_dict["samples avg"]["recall"]
            section_scores["section_avg_f1-score"] = classification_dict["samples avg"]["f1-score"]

            # Combine all scores into a dictionary
            results = {
                "organ_avg_scores": organ_avg_scores,
                "overall_avg_scores": overall_avg_scores,
                "section_scores": section_scores,
            }

            return results

    def run_forward(self,
                    refs,
                    hyps,
                    refs_parsed,
                    hyps_parsed,
                    aligned=True
                    ):

        if aligned:
            all_scores = []
            for ref, hyp in zip(refs_parsed, hyps_parsed):
                # truncate hyp if necessary
                hyp = hyp[:len(ref)]
                # Pad hyps if necessary
                if len(ref) > len(hyp):
                    hyp += ["No findings"] * (len(ref) - len(hyp))
                scores = self.compute_scores(refs=ref, hyps=hyp)
                all_scores.append(scores)

            df = pd.DataFrame(all_scores)
            avg_scores = df.mean().to_dict()
            return avg_scores

        else:
            scores = self.compute_scores(refs=refs,
                                         hyps=hyps,
                                         aligned=False,
                                         parsed_refs=refs_parsed,
                                         parsed_hyps=hyps_parsed)
            return scores

    def compute_scores(self, refs, hyps, aligned=True, parsed_refs=None, parsed_hyps=None):
        if not (isinstance(hyps, list) and isinstance(refs, list)):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")

        scores = {}
        if self.do_bleu:
            scores["bleu"] = self.bleu_scorer(refs, hyps)[0]

        if self.do_rouge:
            for key, scorer in self.rouge_scorers.items():
                scores[key] = scorer(refs, hyps)[0]
        
        if self.do_bertscore:
            scores["bertscore"] = self.bertscore_scorer(refs, hyps)[0]

        if self.do_radgraph:
            radgraph_scores = self.radgraph_scorer(refs=refs, hyps=hyps)
            radgraph_scores = radgraph_scores[0]
            scores["radgraph_simple"] = radgraph_scores[0]
            scores["radgraph_partial"] = radgraph_scores[1]
            scores["radgraph_complete"] = radgraph_scores[2]

        if self.do_green:
            # Compute green score here if needed
            pass

        if self.do_diseases:
            if aligned:
                outputs, _ = self.diseases_model(sentences=refs + hyps)

                refs_preds = outputs[:len(refs)]
                hyps_preds = outputs[len(refs):]

                classification_dict = classification_report(refs_preds, hyps_preds, output_dict=True)
                scores["samples_avg_precision"] = classification_dict["samples avg"]["precision"]
                scores["samples_avg_recall"] = classification_dict["samples avg"]["recall"]
                scores["samples_avg_f1-score"] = classification_dict["samples avg"]["f1-score"]
            else:
                if parsed_refs is None or parsed_hyps is None:
                    raise ValueError("parsed_refs and parsed_hyps must not be None")

                # we have to evaluate section-level, but the model takes utterances as input.
                # We predict on each utterance of the section, then merge labels together to form the section-level pred
                section_level_hyps_pred = []
                section_level_refs_pred = []
                for parsed_hyp, parsed_ref in zip(parsed_hyps, parsed_refs):
                    outputs, _ = self.diseases_model(sentences=parsed_ref + parsed_hyp)

                    refs_preds = outputs[:len(parsed_ref)]
                    hyps_preds = outputs[len(parsed_ref):]

                    merged_refs_preds = np.any(refs_preds, axis=0).astype(int)
                    merged_hyps_preds = np.any(hyps_preds, axis=0).astype(int)

                    section_level_hyps_pred.append(merged_hyps_preds)
                    section_level_refs_pred.append(merged_refs_preds)

                classification_dict = classification_report(section_level_refs_pred,
                                                            section_level_hyps_pred,
                                                            output_dict=True,
                                                            zero_division=0)
                scores["samples_avg_precision"] = classification_dict["samples avg"]["precision"]
                scores["samples_avg_recall"] = classification_dict["samples avg"]["recall"]
                scores["samples_avg_f1-score"] = classification_dict["samples avg"]["f1-score"]
        return scores


def main3():
    refs = [
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Interval resolution of the opacity in the anterior segment of the upper lobe.\n- Subtle persistent opacity at the right lung base laterally, potentially within the right lower lobe.\n\nCardiovascular:\n- Enlarged cardiac silhouette, stable in appearance.\n\nPleura:\n- Posterior costophrenic angles are sharp.\n\nMusculoskeletal and Chest Wall:\n- Osseous and soft tissue structures are unremarkable.',
        'Tubes, Catheters, and Support Devices:\n- Left-sided Automatic Implantable Cardioverter-Defibrillator (AICD) in place\n- Swan Ganz catheter terminating in the right descending pulmonary artery\n- Sternotomy wires intact and aligned\n- Intra-aortic balloon pump previously present has been removed\n\nLungs and Airways:\n- No evidence of pneumothorax\n- Lungs are clear\n\nCardiovascular:\n- Moderate cardiomegaly, stable',
        'Lungs and Airways:\n- The lungs are clear.\n\nCardiovascular:\n- The cardiomediastinal silhouette is within normal limits.\n\nMusculoskeletal and Chest Wall:\n- No acute osseous abnormalities.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No effusion or pneumothorax is present.\n\nCardiovascular:\n- The cardiomediastinal silhouette is normal.\n\nMusculoskeletal and Chest Wall:\n- Osseous structures and soft tissues are unremarkable.',
        'Cardiovascular:\n- Moderate cardiomegaly.\n\nLungs and Airways:\n- Hyperinflated lungs.\n- Biapical scarring without change.\n\nPleura:\n- No pneumothorax or enlarging pleural effusion.\n- Chronic blunting of the right costophrenic angle, which may represent a small effusion or scarring.\n\nMusculoskeletal and Chest Wall:\n- Moderate degenerative changes in the thoracic spine.'
    ]
    hyps = [
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
    ]

    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)

    results = rr.forward(refs=refs, hyps=hyps, section="findings", aligned=True)

    # Access the averaged scores per organ
    organ_avg_scores = results["organ_avg_scores"]
    print("Average Scores per Organ:")
    for organ, scores in organ_avg_scores.items():
        print(f"{organ}: {scores}")

    # Access the overall average scores
    overall_avg_scores = results["overall_avg_scores"]
    print("\nOverall Average Scores:")
    print(overall_avg_scores)

    # Access the section presence scores
    section_scores = results["section_scores"]
    print("\nSection Presence Scores:")
    print(section_scores)
    results = rr.forward(refs=refs, hyps=hyps, section="findings", aligned=False)
    # Access the averaged scores per organ
    organ_avg_scores = results["organ_avg_scores"]
    print("Average Scores per Organ:")
    for organ, scores in organ_avg_scores.items():
        print(f"{organ}: {scores}")

    # Access the overall average scores
    overall_avg_scores = results["overall_avg_scores"]
    print("\nOverall Average Scores:")
    print(overall_avg_scores)

    # Access the section presence scores
    section_scores = results["section_scores"]
    print("\nSection Presence Scores:")
    print(section_scores)

    # Average Scores per Organ:
    # Pleura:: {'radgraph_simple': 0.6041666666666666, 'radgraph_partial': 0.5416666666666666, 'radgraph_complete': 0.5416666666666666, 'bleu': 0.3789528469279699, 'bertscore': 0.6409551687538624, 'rouge1': 0.5930555555555556, 'rouge2': 0.515625, 'rougeL': 0.5930555555555556, 'samples_avg_precision': 0.6875, 'samples_avg_recall': 0.59375, 'samples_avg_f1-score': 0.625}
    # Tubes, Catheters, and Support Devices:: {'radgraph_simple': 0.5, 'radgraph_partial': 0.5, 'radgraph_complete': 0.5, 'bleu': 0.4999999999157178, 'bertscore': 0.5320617146790028, 'rouge1': 0.5069444444444444, 'rouge2': 0.5, 'rougeL': 0.5069444444444444, 'samples_avg_precision': 0.6875, 'samples_avg_recall': 0.6875, 'samples_avg_f1-score': 0.6875}
    # Lungs and Airways:: {'radgraph_simple': 0.5, 'radgraph_partial': 0.5, 'radgraph_complete': 0.5, 'bleu': 0.4999999998311747, 'bertscore': 0.5786798447370529, 'rouge1': 0.5380952380952381, 'rouge2': 0.5, 'rougeL': 0.5380952380952381, 'samples_avg_precision': 0.8, 'samples_avg_recall': 0.8, 'samples_avg_f1-score': 0.8}
    # Musculoskeletal and Chest Wall:: {'radgraph_simple': 0.14285714285714285, 'radgraph_partial': 0.14285714285714285, 'radgraph_complete': 0.14285714285714285, 'bleu': 0.2857142856416828, 'bertscore': 0.3196312678711755, 'rouge1': 0.2857142857142857, 'rouge2': 0.2857142857142857, 'rougeL': 0.2857142857142857, 'samples_avg_precision': 0.2857142857142857, 'samples_avg_recall': 0.2857142857142857, 'samples_avg_f1-score': 0.2857142857142857}
    # Cardiovascular:: {'radgraph_simple': 0.5092592592592592, 'radgraph_partial': 0.4722222222222222, 'radgraph_complete': 0.4444444444444444, 'bleu': 0.22584697511048113, 'bertscore': 0.5879354112678103, 'rouge1': 0.5512820512820513, 'rouge2': 0.4444444444444444, 'rougeL': 0.5512820512820513, 'samples_avg_precision': 0.7777777777777778, 'samples_avg_recall': 0.7777777777777778, 'samples_avg_f1-score': 0.7777777777777778}
    # Hila and Mediastinum:: {'radgraph_simple': 0.5, 'radgraph_partial': 0.5, 'radgraph_complete': 0.5, 'bleu': 0.49999999990881355, 'bertscore': 0.5, 'rouge1': 0.5, 'rouge2': 0.5, 'rougeL': 0.5, 'samples_avg_precision': 0.5, 'samples_avg_recall': 0.5, 'samples_avg_f1-score': 0.5}
    #
    # Overall Average Scores:
    # {'radgraph_simple': 0.4604166666666666, 'radgraph_partial': 0.4395833333333333, 'radgraph_complete': 0.4333333333333333, 'bleu': 0.3766061387175529, 'bertscore': 0.539288105815649, 'rouge1': 0.5028678266178266, 'rouge2': 0.453125, 'rougeL': 0.5028678266178266, 'samples_avg_precision': 0.65625, 'samples_avg_recall': 0.6375, 'samples_avg_f1-score': 0.64375}
    #
    # Section Presence Scores:
    # {'section_avg_precision': 0.8933333333333332, 'section_avg_recall': 0.8916666666666666, 'section_avg_f1-score': 0.8845238095238095}
    # Average Scores per Organ:
    # Pleura:: {'radgraph_simple': 0.6357142857142857, 'radgraph_partial': 0.5857142857142857, 'radgraph_complete': 0.58125, 'bleu': 0.49999999976073317, 'bertscore': 0.6954706236720085, 'rouge1': 0.5981818181818181, 'rouge2': 0.5108695652173914, 'rougeL': 0.5981818181818181, 'samples_avg_precision': 0.75, 'samples_avg_recall': 0.6666666666666666, 'samples_avg_f1-score': 0.6875}
    # Tubes, Catheters, and Support Devices:: {'radgraph_simple': 0.5357142857142857, 'radgraph_partial': 0.5, 'radgraph_complete': 0.5, 'bleu': 0.5108202196971657, 'bertscore': 0.6187150180339813, 'rouge1': 0.5595238095238095, 'rouge2': 0.55, 'rougeL': 0.5595238095238095, 'samples_avg_precision': 0.75, 'samples_avg_recall': 0.625, 'samples_avg_f1-score': 0.6666666666666666}
    # Lungs and Airways:: {'radgraph_simple': 0.54, 'radgraph_partial': 0.54, 'radgraph_complete': 0.54, 'bleu': 0.4999999998475153, 'bertscore': 0.6308901913464069, 'rouge1': 0.585050505050505, 'rouge2': 0.5444444444444445, 'rougeL': 0.585050505050505, 'samples_avg_precision': 0.7333333333333333, 'samples_avg_recall': 0.8, 'samples_avg_f1-score': 0.74}
    # Musculoskeletal and Chest Wall:: {'radgraph_simple': 0.14285714285714285, 'radgraph_partial': 0.14285714285714285, 'radgraph_complete': 0.14285714285714285, 'bleu': 0.2857142856423947, 'bertscore': 0.3161143979855946, 'rouge1': 0.2857142857142857, 'rouge2': 0.2857142857142857, 'rougeL': 0.2857142857142857, 'samples_avg_precision': 0.2857142857142857, 'samples_avg_recall': 0.2857142857142857, 'samples_avg_f1-score': 0.2857142857142857}
    # Cardiovascular:: {'radgraph_simple': 0.5277777777777778, 'radgraph_partial': 0.49074074074074076, 'radgraph_complete': 0.4444444444444444, 'bleu': 0.2258469751124432, 'bertscore': 0.6035888459947374, 'rouge1': 0.5651709401709402, 'rouge2': 0.4444444444444444, 'rougeL': 0.5651709401709402, 'samples_avg_precision': 0.7777777777777778, 'samples_avg_recall': 0.7777777777777778, 'samples_avg_f1-score': 0.7777777777777778}
    # Hila and Mediastinum:: {'radgraph_simple': 0.5, 'radgraph_partial': 0.5, 'radgraph_complete': 0.5, 'bleu': 0.49999999991764293, 'bertscore': 0.5, 'rouge1': 0.5, 'rouge2': 0.5, 'rougeL': 0.5, 'samples_avg_precision': 0.5, 'samples_avg_recall': 0.5, 'samples_avg_f1-score': 0.5}
    #
    # Overall Average Scores:
    # {'radgraph_simple': 0.4844642857142857, 'radgraph_partial': 0.4625595238095238, 'radgraph_complete': 0.45125000000000004, 'bleu': 0.4018975912673429, 'bertscore': 0.5748156843706965, 'rouge1': 0.5240148323898323, 'rouge2': 0.4682850241545894, 'rougeL': 0.5240148323898323, 'samples_avg_precision': 0.6583333333333333, 'samples_avg_recall': 0.6458333333333333, 'samples_avg_f1-score': 0.6391666666666667}
    #
    # Section Presence Scores:
    # {'section_avg_precision': 0.8933333333333332, 'section_avg_recall': 0.8916666666666666, 'section_avg_f1-score': 0.8845238095238095}


def main2():
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)

    hyps = [
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
        '1. The tracheostomy tube is well-positioned without evidence of pneumothorax or pneumomediastinum.\n2. No significant change in the appearance of the heart and lungs when compared to the previous study.',
        '1. Endotracheal tube tip is positioned 3.7 cm above the carina.\n2. Nasogastric tube tip is appropriately located in the stomach.\n3. Stable heart size and mediastinal contours.\n4. Increased left pleural effusion.\n5. Worsening of left retrocardiac consolidation.',
        '1. Improved ventilation of the postoperative right lung.\n2. Expected appearance of the right mediastinal border post-esophagectomy.\n3. Unchanged position of monitoring and support devices.\n4. Slight decrease in soft tissue air collection.\n5. Minimally increased retrocardiac atelectasis.\n6. Normal left lung.',
    ]

    refs = [
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
    ]
    print(rr.forward(refs=refs, hyps=hyps, section="impression", aligned=True))
    # {'radgraph_simple': 0.611111111111111, 'radgraph_partial': 0.611111111111111,
    #  'radgraph_complete': 0.5484848484848485, 'bleu': 0.5805253039653325, 'bertscore': 0.7236699461936951,
    #  'rouge1': 0.6553281392097181, 'rouge2': 0.5955337690631808, 'rougeL': 0.6465562093851568,
    #  'samples_avg_precision': 0.638888888888889, 'samples_avg_recall': 0.638888888888889,
    #  'samples_avg_f1-score': 0.638888888888889}
    # {'radgraph_simple': 0.6095619658119659, 'radgraph_partial': 0.6090788740245262,
    #  'radgraph_complete': 0.5283950617283951, 'bleu': 0.5909341152594585, 'bertscore': 0.7889277338981628,
    #  'rouge1': 0.7133838383838383, 'rouge2': 0.6183064919692035, 'rougeL': 0.6795875420875421,
    #  'samples_avg_precision': 0.8333333333333334, 'samples_avg_recall': 0.6642857142857143,
    #  'samples_avg_f1-score': 0.7194444444444444}

    print(rr.forward(refs=refs, hyps=hyps, section="impression", aligned=False))
    # {'radgraph_simple': 0.5636856368563686, 'radgraph_partial': 0.5636856368563686,
    #  'radgraph_complete': 0.5256410256410257, 'bleu': 0.5843189801389063, 'bertscore': 0.7631320357322693,
    #  'rouge1': 0.6879640921093869, 'rouge2': 0.6173936841103258, 'rougeL': 0.6664468709887289,
    #  'samples_avg_precision': 0.8333333333333334, 'samples_avg_recall': 0.6642857142857143,
    #  'samples_avg_f1-score': 0.7194444444444444}
    stop

    stop


# main2()
# main3()
