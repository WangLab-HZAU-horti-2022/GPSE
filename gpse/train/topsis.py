#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOPSIS model evaluation with optional entropy weighting.

Can be used as a utility module or as a standalone command-line tool.
"""

import pandas as pd
import numpy as np
import argparse
import os


class TOPSISEvaluator:
    """TOPSIS evaluator."""
    
    def __init__(self, logger=None):
        """
        Initialize the TOPSIS evaluator.

        Parameters:
            logger: Logger instance. Uses print output when not provided.
        """
        self.logger = logger
    
    def _log(self, message):
        """Write a log message through the configured logger or print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def entropy_weight_method(self, data):
        """Calculate weights with the entropy weighting method."""
        # Normalize
        P = data / data.sum(axis=0)
        P = P.replace(0, 1e-12)
        n = data.shape[0]
        k = 1.0 / np.log(n)
        E = -k * (P * np.log(P)).sum(axis=0)
        d = 1 - E
        w = d / d.sum()
        return w
    
    def topsis(self, data, weights, criteria_types, min_transform='reciprocal'):
        """Run TOPSIS evaluation."""
        data_proc = data.astype(float).copy()
        weights = np.asarray(weights, dtype=float)
        if data_proc.empty or len(criteria_types) != data_proc.shape[1]:
            raise ValueError("TOPSIS requires nonempty data and a type for every criterion")
        if not np.isfinite(data_proc.to_numpy()).all():
            raise ValueError("TOPSIS criteria must all be finite; no rows may be dropped")
        if any(t not in {"min", "max"} for t in criteria_types):
            raise ValueError("Invalid criterion type")
        if (len(weights) != data_proc.shape[1] or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("TOPSIS weights must be finite, nonnegative, and have positive sum")
        weights = weights / weights.sum()
        for i, ctype in enumerate(criteria_types):
            if ctype == "min":
                col = data_proc.iloc[:, i]
                if (col < 0).any():
                    raise ValueError("Stability/cost criteria must be nonnegative")
                if min_transform == "reciprocal":
                    data_proc.iloc[:, i] = 1.0 / (col + 1e-12)
                elif min_transform == "neglog":
                    # Explicit zero policy: preserve historical epsilon, recorded in output.
                    data_proc.iloc[:, i] = -np.log(col + 1e-12)
                elif min_transform == "minmax_inv":
                    span = col.max() - col.min()
                    data_proc.iloc[:, i] = (col.max() - col) / span if span else 0.0
                else:
                    raise ValueError(f"Unsupported min_transform: {min_transform}")
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.sqrt((data_proc ** 2).sum(axis=0))
        if not np.isfinite(norm).all():
            raise ValueError("Non-finite TOPSIS L2 normalization denominator")
        # Constant criteria contribute equal distances, including an all-zero column.
        data_norm = data_proc / norm.where(norm != 0, 1.0)
        data_weighted = data_norm * weights
        ideal, nadir = data_weighted.max(axis=0), data_weighted.min(axis=0)
        D_pos = np.sqrt(((data_weighted - ideal) ** 2).sum(axis=1))
        D_neg = np.sqrt(((data_weighted - nadir) ** 2).sum(axis=1))
        denominator = D_pos + D_neg
        # Explicit neutral tie when all weighted criteria are constant (also one candidate).
        scores = pd.Series(np.divide(D_neg, denominator, out=np.full(len(data), 0.5),
                                     where=denominator.to_numpy() > 0), index=data.index)
        if not np.isfinite(scores).all():
            raise ValueError("TOPSIS produced non-finite scores")
        return scores, data_proc, data_norm

    def evaluate(self, input_file, output_file, criteria=None, criteria_types=None, 
                 simple_output=None, manual_weights='0.8,0.2', min_transform='neglog',
                 use_entropy_weights=False):
        """
        Run TOPSIS evaluation.

        Parameters:
            input_file: Input CSV file path.
            output_file: Output CSV file path.
            criteria: Evaluation criterion names.
            criteria_types: Criterion types (max/min).
            simple_output: Optional simplified output file path.
            manual_weights: Manual weight string.
            min_transform: Positive transformation method for min-type criteria.
            use_entropy_weights: Whether to use entropy weighting instead of manual weights.

        Returns:
            Processed DataFrame.
        """
        # Set defaults
        if criteria is None:
            criteria = ['cv_mean', 'cv_std_repeat']
        if criteria_types is None:
            criteria_types = ['max', 'min']
        
        # Read data
        df = pd.read_csv(input_file)

        # Validate parameters
        if len(criteria) != len(criteria_types):
            raise ValueError("The number of criteria and criterion types must match")

        if len(criteria) != 2 or criteria[0] != "cv_mean" or criteria[1] not in {
                "cv_std_repeat", "cv_std_within", "cv_std_all_folds"}:
            raise ValueError("Model selection only accepts training-side aggregated CV criteria")
        if criteria_types != ["max", "min"]:
            raise ValueError("CV accuracy must be max and stability must be min")
        required = {"selection_source": "train_only_cv", "aggregation_method": "repeated_cv_v2",
                    "accuracy_metric": "cv_mean", "stability_metric": criteria[1]}
        for key, expected in required.items():
            if key not in df or not (df[key] == expected).all():
                raise ValueError(f"Missing/invalid training-CV provenance: {key}")
        if "Model" not in df or df["Model"].isna().any() or df["Model"].duplicated().any():
            raise ValueError("TOPSIS requires unique model identifiers")
        data = df[criteria].astype(float).copy()
        if use_entropy_weights:
            raise ValueError("CV selection requires fixed manual weights [0.8, 0.2]")
        # Weight configuration
        if use_entropy_weights:
            # Use entropy weighting
            weights = self.entropy_weight_method(data)
            self._log(f"Using entropy weights: {dict(zip(criteria, weights.round(4)))}")
        else:
            # Use manual weights; the default ratio is 8:2
            weights = [float(w.strip()) for w in manual_weights.split(',')]
            if len(weights) != len(criteria):
                raise ValueError("The number of weights must match the number of criteria")
            # Normalize weights
            if not np.allclose(weights, [0.8, 0.2], rtol=0, atol=1e-12):
                raise ValueError("CV selection weights are fixed at [0.8, 0.2]")
            weights = np.array(weights) / sum(weights)
            self._log(f"Using manual weights: {dict(zip(criteria, weights.round(4)))}")

        # Run TOPSIS and return transformed and normalized data
        scores, data_proc, data_norm = self.topsis(data, weights, criteria_types, min_transform)
        df['TOPSIS_Score'] = scores

        # Assign intermediates before sorting so they remain attached to their model.
        # Add intermediate transformation results
        for i, c in enumerate(criteria):
            df[f'{c}_positive'] = data_proc.iloc[:, i]
            df[f'{c}_norm'] = data_norm.iloc[:, i]
            df[f'{c}_weight'] = weights[i]

        df["weight_method"] = "manual"
        df["weights"] = "[0.8,0.2]"
        df["min_transform"] = min_transform
        df["normalization"] = "L2"
        df["zero_sd_policy"] = "epsilon=1e-12"
        df["constant_criterion_policy"] = "equal contribution; all constant => score 0.5, tied rank"
        df = df.sort_values(["TOPSIS_Score", "Model"], ascending=[False, True]).reset_index(drop=True)
        df["TOPSIS_Rank"] = df["TOPSIS_Score"].rank(method="min", ascending=False).astype(int)
        metadata = ["selection_source", "accuracy_metric", "stability_metric", "weight_method", "weights",
                    "aggregation_method", "metric_definition", "code_version", "config_version",
                    "min_transform", "normalization", "zero_sd_policy", "constant_criterion_policy"]
        metadata = [c for c in metadata if c in df]
        # Output required columns for the full result
        output_cols = ['Model'] + criteria + metadata
        output_cols += [f'{c}_positive' for c in criteria]
        output_cols += [f'{c}_norm' for c in criteria]
        output_cols += [f'{c}_weight' for c in criteria]
        output_cols += ['TOPSIS_Score', 'TOPSIS_Rank']
        
        df_out = df[output_cols]
        df_out.to_csv(output_file, index=False)
        self._log(f"Saved TOPSIS evaluation results to: {output_file}")
        self._log(str(df[['Model', 'TOPSIS_Score', 'TOPSIS_Rank']].sort_values('TOPSIS_Rank')))

        # Generate simplified output
        if simple_output:
            simple_cols = ['Model'] + criteria + metadata + ['TOPSIS_Score', 'TOPSIS_Rank']
            df_simple = df[simple_cols]
            df_simple.to_csv(simple_output, index=False)
            self._log(f"Saved simplified TOPSIS results to: {simple_output}")
        
        return df


def entropy_weight_method(data):
    """Calculate weights with entropy weighting; backward-compatible function."""
    evaluator = TOPSISEvaluator()
    return evaluator.entropy_weight_method(data)


def topsis(data, weights, criteria_types, min_transform='reciprocal'):
    """Run TOPSIS evaluation; backward-compatible function."""
    evaluator = TOPSISEvaluator()
    return evaluator.topsis(data, weights, criteria_types, min_transform)


def main():
    parser = argparse.ArgumentParser(description="TOPSIS model evaluation with entropy weighting")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file, such as model_comparison.csv')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--criteria', type=str, default='cv_mean,cv_std_repeat', help='Comma-separated evaluation criterion names')
    parser.add_argument('--criteria_types', type=str, default='max,min', help='Comma-separated criterion types (max/min)')
    parser.add_argument('--simple_output', type=str, default=None, help='Simplified output file name, keeping only raw criteria and TOPSIS results')
    parser.add_argument('--manual_weights', type=str, default='0.8,0.2', help='Manual comma-separated weights; default 0.8,0.2 means accuracy:stability = 8:2')
    parser.add_argument('--min_transform', type=str, default='neglog',
                        choices=['reciprocal', 'neglog', 'minmax_inv'],
                        help='Positive transformation method for min-type criteria; default reciprocal, options neglog and minmax_inv')
    parser.add_argument('--use_entropy_weights', action='store_true',
                        help='Use entropy weighting instead of manual weights')
    args = parser.parse_args()

    # Parse parameters
    criteria = [c.strip() for c in args.criteria.split(',')]
    criteria_types = [c.strip() for c in args.criteria_types.split(',')]

    # Create TOPSIS evaluator and run evaluation
    evaluator = TOPSISEvaluator()
    evaluator.evaluate(
        input_file=args.input,
        output_file=args.output,
        criteria=criteria,
        criteria_types=criteria_types,
        simple_output=args.simple_output,
        manual_weights=args.manual_weights,
        min_transform=args.min_transform,
        use_entropy_weights=args.use_entropy_weights
    )

if __name__ == '__main__':
    main()
