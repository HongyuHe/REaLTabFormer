"""This module contains the implementation for the sampling
algorithms used for tabular and relational data generation.
"""
from __future__ import annotations
from time import perf_counter
from typing import *

if TYPE_CHECKING:
    from .realtabformer import REaLTabFormer
    
import json
import pickle
import networkx as nx
from functools import cache
from collections import OrderedDict
import z3
from rich.pretty import pprint
from IPython.display import display
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import sympy as sp
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DefaultDataCollator, PreTrainedModel

from .data_utils import *
# (
#     INVALID_NUMS_RE,
#     NUMERIC_NA_TOKEN,
#     ModelType,
#     SpecialTokens,
#     decode_column_values,
#     decode_partition_numeric_col,
#     decode_processed_column,
#     fix_multi_decimal,
#     is_datetime_col,
#     is_numeric_col,
#     is_numeric_datetime_col,
#     make_dataset,
#     process_data,
#     to_big_camelcase,
# )
from .rtf_exceptions import SampleEmptyError, SampleEmptyLimitError
from .rtf_validators import ObservationValidator

from collections import defaultdict
# from .anuta_utils import *
# from .anuta_known import *
import anuta
from anuta.constructor import Cidds001, Cicids2017
from anuta.known import cidds_ints, cidds_reals
from anuta.utils import z3evalmap, cidds_flag_map, cidds_proto_map, cidds_ip_map, known_ports

NQ_COL = "_nq_ds_"

cidds_ports_str = [f"{port}pt" for port in cidds_ports]

def get_domain_constraints(varname, evalmap, constructor):
    if constructor.label == 'metadc': 
        #* Domain constraints are already included in the rules.
        return []
    
    domain_constraints = []
    if varname not in constructor.anuta.domains:
        return domain_constraints
    domain = constructor.anuta.domains[varname]
    z3_var = evalmap[varname]
    if domain.bounds:
        #& For numerical vars.
        if type(domain.bounds.lb)==int or type(domain.bounds.ub)==int:
            domain_constraints.append(z3_var >= z3.IntVal(domain.bounds.lb))
            domain_constraints.append(z3_var <= z3.IntVal(domain.bounds.ub))
        else:
            domain_constraints.append(z3_var >= z3.RealVal(domain.bounds.lb))
            domain_constraints.append(z3_var <= z3.RealVal(domain.bounds.ub))
    else:
        #! Adding domain constraints for categorical vars may lead to unsatisfiability for some reason...
        pass
        #& For categorical vars or vars with predefined values.
        # assert len(domain.values)>0
        # if any(type(val)!=np.int64 for val in domain.values):
        #     domain_constraints.append(z3.Or([z3_var==z3.RealVal(val) for val in domain.values]))
        # else:
        #     domain_constraints.append(z3.Or([z3_var==z3.IntVal(val) for val in domain.values]))

    return domain_constraints

def map_to_z3var_value(var_name, value, constructor):
    var_val = None
    match constructor.label:
        case 'cidds':
            #TODO: Wrap the mapping from generated value to rule encoding in a function.
            if var_name=='Flags':
                var_val = z3.IntVal(cidds_flag_map(value))
            elif var_name=='Proto':
                var_val = z3.IntVal(cidds_proto_map(value))
            elif 'ip' in var_name.lower():
                var_val = z3.IntVal(cidds_ip_map(value))
            elif 'pt' in var_name.lower():
                value = int(value[:-2]) if value[-2:] == 'pt' else int(value)
                var_val = z3.IntVal(value)
            else:
                try:
                    if '.' in value:
                        var_val = z3.RealVal(float(value))
                    elif '_' not in value:
                        var_val = z3.IntVal(int(value))
                except ValueError:
                    var_val = value
        case 'cicids':
            domain = constructor.anuta.domains[var_name]
            if var_name == 'Protocol':
                var_val = z3.IntVal(int(value))
            elif type(domain.bounds.lb)==float or type(domain.bounds.ub)==float:
                var_val = z3.RealVal(float(value))
            else:
                var_val = z3.IntVal(int(value))
        case _:
            raise ValueError(f"Unknown dataset: {constructor.label}")
    
    assert var_val is not None, f"Rule value not mapped for {var_name}={value}"
    return var_val

class REaLSampler:
    def __init__(
        self,
        model_type: str,
        model: PreTrainedModel,
        vocab: Dict,
        processed_columns: List,
        max_length: int,
        col_size: int,
        col_idx_ids: Dict,
        columns: List,
        datetime_columns: List,
        column_dtypes: Dict,
        column_has_missing: Dict,
        drop_na_cols: List,
        col_transform_data: Dict,
        random_state: Optional[int] = 1029,
        device="cuda",
    ) -> None:
        self.model_type = model_type
        self.vocab = vocab
        self.processed_columns = processed_columns
        self.col_size = col_size  # relational_col_size or tabular_col_size
        self.col_idx_ids = col_idx_ids
        self.max_length = max_length  # relational_max_length or tabular_max_length

        self.columns = columns
        self.datetime_columns = datetime_columns

        self.column_dtypes = column_dtypes
        self.column_has_missing = column_has_missing
        self.drop_na_cols = drop_na_cols

        self.col_transform_data = col_transform_data

        self.random_state = random_state

        self.device = torch.device(device)

        if model.device != self.device:
            self.model = model.to(self.device)
        else:
            self.model = model

        self.invalid_gen_samples = 0
        self.total_gen_samples = 0

        # Set the model to eval mode
        self.model.eval()

    def _prefix_allowed_tokens_fn(self, batch_id, input_ids) -> List:
        # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.prefix_allowed_tokens_fn
        raise NotImplementedError

    def _convert_to_table(self, synth_df: pd.DataFrame) -> pd.DataFrame:
        # Perform additional standardization
        # processing.
        synth_df = synth_df[sorted(synth_df.columns)]
        synth_df.columns = synth_df.columns.map(decode_processed_column)

        # Order based on the original data columns.
        try:
            synth_df = synth_df[self.columns]
        except KeyError:
            pass

        for col in self.datetime_columns:
            # Attempt to transform datetime data
            # from the encoded timestamp into datetime.
            try:
                # Add the `mean_date` that we have subtracted during
                # fitting of the data.
                series = synth_df[col].copy()
                series[series.notna()] += self.col_transform_data[col].get(
                    "mean_date", 0
                )

                # Multiply by 1e9 since we divided by this
                # the original timestamp in the processing step.
                synth_df[col] = pd.to_datetime(series * 1e9)
            except Exception:
                pass

        # Attempt to cast the synthetic data to
        # the original data types.
        valid_idx = set(range(len(synth_df)))
        _synth_df = []
        for c, d in self.column_dtypes.items():
            try:
                series = synth_df[c]

                if pd.api.types.is_numeric_dtype(d):
                    if self.column_has_missing[c]:
                        # If original data type is numeric,
                        # attempt to replace the <NA> token
                        # with NaN for casting to work.
                        series = series.replace("<NA>", "NaN")

                    if series.dtype == "object":
                        # Explicitly unify the data type
                        # in case we have mixed types in the series.
                        series = series.astype(str)

                        nan_idx = None
                        if self.column_has_missing[c]:
                            # Values where nan was generated first is treated
                            # as a valid nan value.
                            nan_idx = series.str.startswith(NUMERIC_NA_TOKEN)
                            series.loc[nan_idx] = "NaN"

                        # Removed indices of invalid values. Invalid values
                        # are those where nan was generated somewhere after
                        # the first token set for the variable or that any
                        # non-numeric value is present.
                        re_invalid_pattern = INVALID_NUMS_RE

                        # The initially changed "NaN" above will be
                        # included by the pattern. Make sure we add indices
                        # back to the valid_idx list.
                        invalid_idx = series.str.contains(
                            re_invalid_pattern, regex=True
                        )
                        valid_idx = valid_idx.difference(np.where(invalid_idx)[0])

                        if nan_idx is not None:
                            valid_idx = valid_idx.union(np.where(nan_idx)[0])

                        # Temporarily set them to NaN, but we will remove these
                        # observations later.
                        series.loc[invalid_idx] = "NaN"
                elif pd.api.types.is_object_dtype(d):
                    # In case the values in this column is "integral"
                    # but are cast as object in the real data, let's try
                    # to convert them first to int before casting to the
                    # correct object type.
                    # Not doing this will incorrectly generate "float"-looking
                    # data for object types, but are integral, in the
                    # fb-comments dataset.
                    try:
                        series = series.astype(pd.Int64Dtype()).astype(str)
                    except TypeError:
                        pass
                    except ValueError:
                        pass

                if series.dtype != d:
                    # To speed up a bit, only cast if the dtypes
                    # is not the same as the expected type.
                    series = series.astype(d)

                _synth_df.append(series)
            except Exception:  # nolint
                _synth_df.append(synth_df[c])

        synth_df = pd.concat(_synth_df, axis=1)

        # We expect columns that have no missing values in the
        # training data should have no <NA> in them.
        with_missing_cols = [
            col for col, missing in self.column_has_missing.items() if missing
        ]
        if with_missing_cols:
            synth_df[with_missing_cols] = (
                synth_df[with_missing_cols]
                .replace(f".*{NUMERIC_NA_TOKEN}+.*", "NaN", regex=True)
                .replace("<NA>", "NaN")
                .replace("NaN", np.nan)
            )

        if len(valid_idx) > 0:
            synth_df = synth_df.iloc[sorted(valid_idx)]

        return synth_df

    def _generate(
        self,
        device: torch.device,
        as_numpy: Optional[bool] = True,
        constrain_tokens_gen: Optional[bool] = True,
        **generate_kwargs,
    ) -> Union[torch.tensor, np.ndarray]:
        # This leverages the generic interface of HuggingFace transformer models' `.generate` method.
        # Refer to the transformers documentation for valid arguments to `generate_kwargs`.
        self.model.eval()

        if constrain_tokens_gen:
            #TODO: This might be the place to constrain the allowed next tokens?
            generate_kwargs["prefix_allowed_tokens_fn"] = self._prefix_allowed_tokens_fn

        vocab = (
            self.vocab
            if self.model_type == ModelType.tabular
            else self.vocab["decoder"]
        )

        # Make sure that the [RMASK] token will never be generated.
        RMASK_ID = vocab["token2id"][SpecialTokens.RMASK]
        if generate_kwargs["suppress_tokens"] is None:
            #TODO: Or here? // "A list of tokens that will be supressed at generation. The SupressTokens logit processor will set their log probs to -inf so that they are not sampled."
            #? Will the distribution be normalized?
            generate_kwargs["suppress_tokens"] = [RMASK_ID]
        else:
            generate_kwargs["suppress_tokens"].append(RMASK_ID)

        if "bos_token_id" not in generate_kwargs:
            generate_kwargs["bos_token_id"] = vocab["token2id"][SpecialTokens.BOS]

        if "pad_token_id" not in generate_kwargs:
            generate_kwargs["pad_token_id"] = vocab["token2id"][SpecialTokens.PAD]

        if "eos_token_id" not in generate_kwargs:
            generate_kwargs["eos_token_id"] = vocab["token2id"][SpecialTokens.EOS]

        #TODO: Customize `inputs` to pass part of the example as the "prompt"
        # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate.inputs
        _samples = self.model.generate(**generate_kwargs)

        if as_numpy:
            if device == torch.device("cpu"):
                _samples = _samples.numpy()
            else:
                _samples = _samples.cpu().numpy()

        return _samples

    def _validate_synth_sample(self, synth_sample: pd.DataFrame) -> pd.DataFrame:
        # Validate data
        valid_mask = []

        # Validate that the generated value for a column is correct.
        # Use the column identifier as basis for the validation.
        # Is this useful when we actually filter the vocabulary
        # during generation???
        # Let's try removing this for now and see what happens... XD
        # We remove this because the operations here are expensive, and
        # slow down the sampling.
        for col in synth_sample.columns:
            valid_mask.append(synth_sample[col].str.startswith(col))

        valid_mask = pd.concat(valid_mask, axis=1).all(axis=1)
        synth_sample = synth_sample.loc[valid_mask]

        if synth_sample.empty:
            # Handle this exception in the sampling function.
            raise SampleEmptyError(in_size=len(valid_mask))

        return synth_sample

    def _recover_data_values(self, synth_sample: pd.DataFrame) -> pd.DataFrame:
        processed_columns = pd.Index(self.processed_columns)
        numeric_datetime_cols = processed_columns[
            processed_columns.map(is_numeric_datetime_col)
        ]

        _tmp_synth_df = []
        _tmp_synth_cols = []

        # Get the actual column name for numerically
        # transformed data.
        numeric_col_group = numeric_datetime_cols.groupby(
            numeric_datetime_cols.map(decode_partition_numeric_col)
        )

        for col, c_group in numeric_col_group.items():
            # Aggregate the partitioned data for the actual column.
            group_series = synth_sample[c_group].sum(axis=1)

            if group_series.dtype == "object":
                # This automatically casts into numeric type if all values are
                # valid numbers.
                group_series = group_series.str.lstrip("0")

                # If all values are zeros for a group, then it will be left
                # empty. Let's explicitly set the zero value back.
                # TODO: review other use cases that this may not be correct.
                # For example, zero may not be in the domain of the
                # variable `CompetitionDistance` in the Rossmann dataset.
                group_series[group_series == ""] = 0

                # If the data is still an object type, try to fix potential
                # errors.
                if group_series.dtype == "object":
                    if is_numeric_col(col):
                        try:
                            # This usually happens when the decimal point was generated
                            # multiple times in the value. This simply removes the succeeding
                            # occurence of the decimal point.
                            group_series = (
                                group_series.apply(fix_multi_decimal)
                                .astype(float)
                                .fillna(pd.NA)
                            )
                        except Exception:
                            pass
                    elif is_datetime_col(col):
                        # We expect that timestamp data is represented fully
                        # by numbers. Just remove all non-numeric characters.
                        # This may introduce invalid values somehow, but validators
                        # can later be implemented to remove these.
                        group_series = (
                            group_series.str.replace("[^0-9]", "", regex=True)
                            .map(lambda x: int(x) if x else None)
                            .fillna(pd.NA)
                        )
                    else:
                        raise ValueError(f"Unknown column dtype for {col}...")

                if is_numeric_col(col):
                    try:
                        # Try to force convert values to Int64Dtype
                        group_series = group_series.astype(pd.Int64Dtype())
                    except TypeError:
                        pass
                    except ValueError:
                        # Example:
                        # ValueError: invalid literal for int() with base 10: '1271.0942'
                        pass

            _tmp_synth_df.append(group_series)
            _tmp_synth_cols.append(col)

        # Add data for categorical columns
        for col in synth_sample.columns:
            if col in numeric_datetime_cols:
                continue

            _tmp_synth_df.append(synth_sample[col])
            _tmp_synth_cols.append(col)

        synth_df: pd.DataFrame = pd.concat(_tmp_synth_df, axis=1).reset_index(
            drop="index"
        )
        synth_df.columns = _tmp_synth_cols
        synth_df.index = synth_sample.index

        return synth_df

    def _processes_sample(
        self,
        sample_outputs: np.ndarray,
        vocab: Dict,
        relate_ids: Optional[List[Any]] = None,
        validator: Optional[ObservationValidator] = None,
    ) -> pd.DataFrame:
        assert isinstance(sample_outputs, np.ndarray)

        def _decode_tokens(s):
            # No need to remove [BOS] and [EOS] tokens
            # here, it will be handled later.
            return [vocab["id2token"][i] for i in s]

        if self.model_type == ModelType.tabular:
            # Slice to remove the [BOS] and [EOS] tokens
            synth_sample = pd.DataFrame(
                [_decode_tokens(s)[1:-1] for s in sample_outputs],
                columns=self.processed_columns,
            )
        else:
            assert relate_ids is not None
            _samples = [_decode_tokens(s) for s in sample_outputs]

            # Unpack the tokens and remove any special tokens.
            # Also perform segmentation of observations for
            # the relational model.
            group_ids = []
            samples = []
            for ix, (rel_id, dg) in enumerate(zip(relate_ids, _samples)):
                group = []
                ind: List[str] = []
                for token_ix, v in enumerate(dg):
                    if v in [SpecialTokens.BMEM, SpecialTokens.EOS]:
                        if len(ind) > 0:
                            # Review later whether we should void
                            # the entire generation process for this
                            # input or not. For now, let's just
                            # throw this particular observation that didn't
                            # satisfy the expected col_size.
                            if len(ind) == self.col_size:
                                group.append(ind)
                            else:
                                logging.warning(
                                    f"Discarding this observation for input index:{ix} with an invalid number of columns: {len(ind)}."
                                )
                            ind = []
                        if (v == SpecialTokens.EOS) and (token_ix > 0):
                            break
                    elif v in [SpecialTokens.BOS, SpecialTokens.EMEM]:
                        continue
                    elif v == SpecialTokens.PAD:
                        # This should not go here, but putting this just in case.
                        break
                    else:
                        ind.append(v)

                group_ids.extend([rel_id] * len(group))
                samples.extend(group)

            # Create a unique index for observations
            # that are supposed to be generated by the
            # same input data.
            synth_sample = pd.DataFrame(
                samples, columns=self.processed_columns, index=group_ids
            )

        # Initial check for an empty sample frame.
        if synth_sample.empty:
            # Handle this exception in the sampling function.
            raise SampleEmptyError(in_size=len(sample_outputs))

        # # Is this useful when we actually filter the vocabulary
        # # during generation???
        # # Let's try removing this for now and see what happens... XD
        # # We remove this because the operations here are expensive, and
        # # slow down the sampling.
        # synth_sample = self._validate_synth_sample(synth_sample)

        # Extract the values
        for col in synth_sample.columns:
            # Filter only columns that we have explicitly processed.
            # Since the column values have been previously validated
            # to only contain values that match the column, then it
            # is safe to just check the first row for this contraint.
            if synth_sample[col].iloc[0].startswith(col):
                synth_sample[col] = decode_column_values(synth_sample[col])

        synth_df = self._recover_data_values(synth_sample)
        logging.info(f"Generation stats: {synth_df.shape[0]}")

        synth_df = self._convert_to_table(synth_df)
        synth_df = self._validate_missing(synth_df)
        synth_df = self._validate_data(synth_df, validator)

        if synth_df.empty:
            # Handle this exception in the sampling function.
            raise SampleEmptyError(in_size=len(sample_outputs))

        return synth_df

    def _validate_data(
        self, synth_df: pd.DataFrame, validator: Optional[ObservationValidator] = None
    ) -> pd.DataFrame:
        if validator is not None:
            synth_df = synth_df.loc[validator.validate_df(synth_df)]

        return synth_df

    def _validate_missing(self, synth_df: pd.DataFrame) -> pd.DataFrame:
        # Drop the rows where any one of the columns that should not have
        # a missing value have at least one.
        return synth_df.dropna(subset=self.drop_na_cols)


class TabularSampler(REaLSampler):
    """Sampler class for tabular data generation."""

    def __init__(
        self,
        model_type: str,
        model: PreTrainedModel,
        vocab: Dict,
        processed_columns: List,
        max_length: int,
        col_size: int,
        col_idx_ids: Dict,
        columns: List,
        datetime_columns: List,
        column_dtypes: Dict,
        column_has_missing: Dict,
        drop_na_cols: List,
        col_transform_data: Dict,
        random_state: Optional[int] = 1029,
        
        col_start_pos: Optional[List[int]] = None,
        varnames: List[str] = [],
        relevant_rules: Optional[Dict[str, List[z3.ExprRef]]] = None,
        constructor: Optional[anuta.constructor.Constructor ] = None,
        evalmap: Optional[Dict[str, z3.ExprRef]] = None,
        col_ignore: Optional[List[int]] = None,
        device="cuda",
    ) -> None:
        super().__init__(
            model_type,
            model,
            vocab,
            processed_columns,
            max_length,
            col_size,
            col_idx_ids,
            columns,
            datetime_columns,
            column_dtypes,
            column_has_missing,
            drop_na_cols,
            col_transform_data,
            random_state,
            device,
        )

        self.output_vocab = self.vocab
        #* Start indexes of original columns from processed columns
        self.col_start_pos = col_start_pos
        self.relevant_rules = relevant_rules
        self.prefix_rules = []
        self._generation_cache = {}
        self.constructor = constructor
        self.evalmap = evalmap
        self._numeric_gen = {}
        self.col_ignore = col_ignore
        self.num_invalid_tokens = 0
        self._sample_validities: List[bool] = []
        
        self.varnames = varnames
        
        with open("/home/hh1789/Notebooks/data/cidds_trie_prefix_freeflags.pkl", 'rb') as f:
            self.trie: nx.DiGraph= pickle.load(f)
            print(f"Loaded trie with {len(self.trie.nodes())} nodes and {len(self.trie.edges())} edges.")

    @staticmethod
    def sampler_from_model(rtf_model: 'REaLTabFormer', dataset: str='cidds', device: str = "cuda"):
        device = torch.device(device)

        assert rtf_model.tabular_max_length is not None
        assert rtf_model.tabular_col_size is not None
        assert rtf_model.col_transform_data is not None
        
        #& Extract the start positions of the original columns from the mangled column names.
        processed_cols = rtf_model.vocab['column_token_ids'].keys()
        col_start_pos = []
        cur_col = None
        for i, col in enumerate(processed_cols):
            col = col.split('___')[-1]
            col = ''.join(col.split('_')[:-1]) if '_' in col else col
            if col != cur_col:
                col_start_pos.append(i)
                cur_col = col
        #* Include the end position + 1 in order to generate the last column (its previous col).
        col_start_pos.append(len(processed_cols))
        # assert len(col_start_pos) == len(rtf_model.columns)+1, f"{len(col_start_pos)=} ≠ {len(rtf_model.columns)=}"
        print(f"{col_start_pos=}")
        
        #& Populate prefix rules.
        rulepath = ''
        evalmap = z3evalmap
        constructor = None
        col_todrop = []
        col_ignore = []
        varnames = []
        match dataset:
            case 'cidds':
                rulepath = "/home/hh1789/Projects/REaLTabFormer/rules/learned_cidds_8192_checked.pl"
                datapath = "/scratch/gpfs/hh1789/data/cidds_wk3_all.csv"
                constructor = Cidds001(datapath)
                col_todrop = ['Flows']
                varnames = [to_big_camelcase(col) for col in rtf_model.columns]
                # rtf_model.column_dtypes = {
                #     to_big_camelcase(col): dtype for col, dtype in rtf_model.column_dtypes.items()
                # }
            case 'cicids':
                rulepath = "/home/hh1789/Projects/REaLTabFormer/rules/learned_cicids_8192_checked.pl"
                datapath = "/scratch/gpfs/hh1789/data/cicids_monday_all.csv"
                constructor = Cicids2017(datapath)
                col_todrop = ['Down_Up_Ratio', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size', 
                            'Fwd_Avg_Bytes_Bulk', 'Fwd_Avg_Packets_Bulk', 'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes_Bulk', 
                            'Bwd_Avg_Packets_Bulk', 'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes', 
                            'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes', 'Init_Win_bytes_fwd', 'Init_Win_bytes_bwd', 
                            'act_data_pkt_fwd', 'min_seg_size_fwd', 'Source_Port', 'Destination_Port', ]
                col_todrop = [to_big_camelcase(col, '_') for col in col_todrop]
                varnames = [to_big_camelcase(col, '_') for col in rtf_model.columns]
                # rtf_model.column_dtypes = {
                #     to_big_camelcase(col, '_'): dtype for col, dtype in rtf_model.column_dtypes.items()
                # }
            case 'metadc':
                rulepath = '/home/hh1789/Projects/REaLTabFormer/rules/lgbm_metadc_all.pl'
                datapath = '/scratch/gpfs/hh1789/data/metadc_train.csv'
                col_todrop = ['rackid', 'hostid']
                varnames = rtf_model.columns
            case _:
                raise ValueError(f"Unknown dataset: {dataset}")

        for col in col_todrop:
            if col in rtf_model.columns:
                col_ignore.append(rtf_model.columns.index(col))
                # rtf_model.columns.remove(col)
        rules_sp = []
        with open(f"{rulepath}", 'r') as f:
            for i, line in enumerate(f):
                expr: sp.Expr = sp.sympify(line.strip())
                rules_sp.append(expr)
                print(f"Loaded # of rules:\t{i+1}", end='\r')
        
        #* First, complete the evalmap for sp to z3 conversion.
        for varname in varnames:
            match dataset:
                case 'cidds':
                    #* Update the evalmap with vars.
                    if varname in cidds_ints:
                        evalmap[varname] = z3.Int(varname)
                    elif varname in cidds_reals:
                        evalmap[varname] = z3.Real(varname)
                case 'cicids':
                    if varname in constructor.anuta.domains:
                        domain = constructor.anuta.domains[varname]
                        if varname == 'Protocol':
                            evalmap[varname] = z3.Int(varname)
                        elif type(domain.bounds.lb)==float or type(domain.bounds.ub)==float:
                            evalmap[varname] = z3.Real(varname)
                        else:
                            evalmap[varname] = z3.Int(varname)
                case 'metadc':
                    evalmap[varname] = z3.Int(varname)
                case _:
                    raise ValueError(f"Unknown dataset: {dataset}")
        
        relevant_rules = defaultdict(list)
        for i, varname in enumerate(varnames):
            relevant = []
            filtered_relevant = []
            prefix_vars = set(sp.symbols([name for name in varnames[: i]]))
            
            cur_var = sp.symbols(varname)
            included_vars = prefix_vars | {cur_var}
            for rule in rules_sp:
                rule_vars = rule.free_symbols
                num_vars = len(rule_vars)
                
                # #& All connected rules.
                # if len(variables & included_vars) > 0:
                #     relevant.append(rule)
                #     #* Accumulate vars from included rules.
                #     included_vars |= variables
                
                # #& Related rules.
                # if len(variables & (prefix_vars | {cur_var})) > 0:
                #     #* As long as the rule contains ≥1 variable from the current and/or one of the prefix vars.
                #     relevant.append(rule)

                #& Prefix rules only.
                if cur_var in rule_vars and len(rule_vars & (prefix_vars | {cur_var})) == num_vars:
                    #* rule has to contain current var AND the rest of vars are all prefixes
                    relevant.append(rule)
            print(f"Found {len(relevant)} relevant rules for {varname}.")
            
            #* Filter redundant rules
            for rule in relevant:
                if isinstance(rule, sp.Implies):
                    antecedent, consequent = rule.args
                    if isinstance(antecedent, sp.And):
                        p1, p2 = antecedent.args
                        # pprint(rule)
                        # pprint([p1.free_symbols, p2.free_symbols])
                        #* Tautology: (X=x1 ∧ X=x2) ⇒ ...
                        if p1.free_symbols == p2.free_symbols:
                            # pprint(rule)
                            continue
                filtered_relevant.append(rule)
            filtered_relevant = sorted(filtered_relevant, key=lambda r: str(r))
            # print(f"Filtered to {len(filtered_relevant)} relevant rules for {varname}.")
            
            # coalesced_rules = coalesce(filtered_relevant)
            # print(f"Coalesced to {len(coalesced_rules)} rules for {varname}.")
            # # if varname == 'SrcPt':
            # #     for rule in coalesced_rules:
            # #         # display(rule)
            # #         print(rule)
            
            # rules_z3 = [eval(str(rule), evalmap) for rule in coalesced_rules]
            # print(str(rule))
            display(eval(str(rule), evalmap))
            rules_z3 = [eval(str(rule), evalmap) for rule in filtered_relevant]
            relevant_rules[varname] = rules_z3
        
        for varname, rules in relevant_rules.items():
            checked_rules = set()
            for rule in rules:
                isvalid = True
                solver = z3.Solver()
                rule = z3.simplify(rule)
                solver.add(~rule)
                if solver.check() == z3.unsat:
                    isvalid = False
                    # print(f"Tautology:")
                    # display(rule)
                
                solver = z3.Solver()
                solver.add(rule)
                if solver.check() == z3.unsat:
                    isvalid = False
                    # print(f"Contradiction:")
                    # display(rule)

                if isvalid:
                    checked_rules.add(rule)
            print(f"{varname}: {len(rules)-len(checked_rules)}/{len(rules)} invalid rules.")
            
            checked_rules = list(checked_rules)
            domain_constraints = get_domain_constraints(varname, evalmap, constructor)
            #* Constraint the feasible tokens within the var's domain.
            checked_rules.extend(domain_constraints)
            #* Combine all rules into a single theorem.
            combined_rule = z3.And(checked_rules)
            relevant_rules[varname] = combined_rule

        return TabularSampler(
            model_type=rtf_model.model_type,
            model=rtf_model.model,
            vocab=rtf_model.vocab,
            processed_columns=rtf_model.processed_columns,
            max_length=rtf_model.tabular_max_length,
            col_size=rtf_model.tabular_col_size,
            col_idx_ids=rtf_model.col_idx_ids,
            columns=rtf_model.columns,
            datetime_columns=rtf_model.datetime_columns,
            column_dtypes=rtf_model.column_dtypes,
            column_has_missing=rtf_model.column_has_missing,
            drop_na_cols=rtf_model.drop_na_cols,
            col_transform_data=rtf_model.col_transform_data,
            random_state=rtf_model.random_state,
            
            col_start_pos=col_start_pos,
            varnames=varnames,
            relevant_rules=relevant_rules,
            constructor=constructor,
            evalmap=evalmap,
            col_ignore=col_ignore,
            device=device,
        )
    
    # @cache
    def _is_token_valid(self, token: int, nxt_var_name: str, generated: Tuple) -> bool:
        value = self.vocab["id2token"][token].split(SPECIAL_COL_SEP)[-1]
        #* Domain knowledge: Only check known ports.
        if 'Pt' in nxt_var_name and value not in cidds_ports_str:
            return True
        
        z3var = self.evalmap[nxt_var_name]
        z3val = map_to_z3var_value(nxt_var_name, value, self.constructor)
        
        #& Using AND-combined rules.
        relevant_rule = self.relevant_rules[nxt_var_name]
        if not relevant_rule: 
            #* No rules to check for this variable.
            return True
        substituted = z3.simplify(z3.substitute(relevant_rule, (z3var, z3val), *generated))
        s = z3.Solver()
        s.add(substituted)
        return True if s.check() == z3.sat else False
    
    # @cache
    def _solve_for_minmax(self, nxt_var_name: str, generated: Tuple) -> Tuple[float, float]:
        relevant_rule = self.relevant_rules[nxt_var_name]
        z3var = self.evalmap[nxt_var_name]
        
        #& Using AND-combined rules.
        opt = z3.Optimize()
        substituted = z3.simplify(z3.substitute(relevant_rule, *generated))
        opt.add(substituted)
        # for rule in substituted_rules:
        #     opt.add(rule)
        lb = opt.minimize(z3var)
        if opt.check() == z3.sat:
            lb = lb.value()
            if isinstance(lb, z3.RatNumRef):
                lb = float(lb.as_fraction())
            elif isinstance(lb, z3.IntNumRef):
                lb = lb.as_long()
            else:
                assert isinstance(lb, z3.ArithRef), f"Unknown {type(lb)=} for {lb=}"
                # print(f"z3.ArithRef: {lb=}")
                domain = self.constructor.anuta.domains[nxt_var_name]
                lb = domain.bounds.lb
                # for rule in substituted_rules:
                #     display(rule)
        else:
            print(f"[Optm] Can't obtain logit lower bound for {z3var}")
            domain = self.constructor.anuta.domains[nxt_var_name]
            lb = domain.bounds.lb
            # for rule in substituted_rules:
            #     display(rule)
        
        opt = z3.Optimize()
        # substituted = z3.simplify(z3.substitute(relevant_rule, *generated))
        opt.add(substituted)
        # for rule in substituted_rules:
        #     opt.add(rule)
        ub = opt.maximize(z3var)
        if opt.check() == z3.sat:
            ub = ub.value()
            if isinstance(ub, z3.RatNumRef):
                ub = float(ub.as_fraction())
            elif isinstance(ub, z3.IntNumRef):
                ub = ub.as_long()
            else:
                assert isinstance(lb, z3.ArithRef), f"Unknown {type(ub)=} for {ub=}"
                # print(f"z3.ArithRef: {ub=}")
                domain = self.constructor.anuta.domains[nxt_var_name]
                ub = domain.bounds.ub
                # for rule in substituted_rules:
                #     display(rule)
        else:
            print(f"[Optm] Can't obtain logit upper bound for {z3var}")
            domain = self.constructor.anuta.domains[nxt_var_name]
            ub = domain.bounds.ub
            # for rule in substituted_rules:
            #     display(rule)
        return lb, ub
    
    def _prefix_allowed_tokens_fn(self, batch_id, input_ids) -> List:
        # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.prefix_allowed_tokens_fn
        # For the tabular data, len(input_ids) == 1 -> [BOS]
        # Subtract by 1 since the first valid token has index zero in
        # col_idx_ids while the input_ids already contains the [BOS] token.
        nxt_token_idx = len(input_ids) - 1
        #* Current token idx is len(input_ids) - 2
        nxt_col_idx = None
        if nxt_token_idx == 0:
            #! Use OrderedDict to preserve the order of generated values for DP caching.
            self._generation_cache[batch_id] = {'incomplete': '', 'generated': OrderedDict()}
            self._numeric_gen[batch_id] = {'started': False}
        #* `len(input_ids)` is the number of tokens generated so far.
        #* It's also the column index of the next column to be generated (if all cols are categorical).
        #* Use this as the key to get the valid tokens for the next column.
        col_valid_tokenids = self.col_idx_ids.get(nxt_token_idx, [self.vocab["token2id"][SpecialTokens.EOS]])
        
        col_finished = False
        generated_val = self.vocab["id2token"][input_ids[-1].item()].split(SPECIAL_COL_SEP)[-1]
        if nxt_token_idx in self.col_start_pos and nxt_token_idx > 0: 
            complete_value = self._generation_cache[batch_id]['incomplete'] + generated_val
            #* Clear the cache for the next column generation.
            self._generation_cache[batch_id]['incomplete'] = ''
            nxt_col_idx = self.col_start_pos.index(nxt_token_idx)
            col_idx = nxt_col_idx - 1 #* The last column generated (not the next column to be generated)
            if col_idx in self.col_ignore:
                return col_valid_tokenids
            col_name = self.columns[col_idx]
            var_name = self.varnames[col_idx]
            #* Map the generated value to rule encoding
            #! There's also a mismatch between the generated value and the value range of the rules.
            z3val = map_to_z3var_value(var_name, complete_value, self.constructor)
                        
            self._generation_cache[batch_id]['generated'][var_name] = z3val
            #TODO: Update trie node.
            parent_id = self.parent_node[batch_id]
            self.parent_node[batch_id] = f"{parent_id}->{var_name}" \
                if self.column_dtypes[col_name]!='object' \
                    else f"{parent_id}->{var_name}::{int(z3val.as_long())}"
            #* Deal with private ports.
            if 'Pt' in var_name and complete_value not in cidds_ports_str:
                self.parent_node[batch_id] = f"{parent_id}->{var_name}::60000"
            # print(f"\tCurrent trie node: {self.parent_node[batch_id].split('->')[-1]}")
            self._numeric_gen[batch_id] = {'started': False}
            # assert len(self._generation_cache[batch_id]['generated']) == col_idx + 1
            #* The current column has been fully generated.
            if nxt_token_idx != self.col_start_pos[-1]:
                #* If it's not the last column, generate the next column.
                col_finished = True 

        elif nxt_token_idx > 0:
            #* Cache the generated (incomplete) value for the current column.
            self._generation_cache[batch_id]['incomplete'] += generated_val
        
        if col_finished:
            #* Check if the generated value is valid. If not, mark this sample and skip its checks.
            isvalidsample = self._sample_validities[batch_id]
            #* Check the current var (for categorical vars, we can determine its validity during generation.
            #*  But for numeric vars, we need to check the generated value against the rules after 
            #*  complete value has been generated).
            if isvalidsample: # and self._numeric_gen[batch_id]['started']: #! Only check numeric vars when using the trie.
                relevant_rule = self.relevant_rules[var_name]
                generated = [(self.evalmap[name], val) for name, val 
                            in self._generation_cache[batch_id]['generated'].items()
                            if name in self.evalmap]
                s = z3.Solver()
                try:
                    substituted = z3.substitute(relevant_rule, *generated)
                except z3.Z3Exception as e:
                    print(f"!!! Z3Exception: {e} for {var_name=} {relevant_rule=}")
                    raise e
                s.add(substituted)
                if s.check() == z3.unsat:
                    isvalidsample = False
                    print(f"!!! Sample {batch_id=} is invalid.")
                self._sample_validities[batch_id] = isvalidsample
            else:
                #! Let go when it's already invalid.
                return col_valid_tokenids
            
            assert nxt_col_idx is not None
            nxt_var_name = self.varnames[nxt_col_idx]
            # relevant_rule = self.relevant_rules[nxt_var_name]
            # print(f"\t{len(relevant_rules)=} for {nxt_var_name}")
            
            if nxt_var_name not in self.evalmap: 
                return col_valid_tokenids

            #TODO: Figure out why 'Flows' still occurs.
            if nxt_var_name not in self.constructor.anuta.domains:
                return col_valid_tokenids
            
            domain = self.constructor.anuta.domains[nxt_var_name]
            # generated = [(self.evalmap[name], val) for name, val 
            #              in self._generation_cache[batch_id]['generated'].items()
            #              if name in self.evalmap]
            generated = [] #* No need when using the trie.
            valid_tokens = []
            if domain.values is not None and len(domain.values) > 0:
                #& Categorical variable or a variable with predefined values.
                nxt_values = [self.trie.nodes[nodeid]['value']
                              for nodeid in self.trie.successors(self.parent_node[batch_id])]
                assert len(nxt_values) > 0, f"No child nodes for {self.parent_node[batch_id]}"
                for tokenid in col_valid_tokenids:
                    # isvalid = self._is_token_valid(token, nxt_var_name, tuple(generated))
                    
                    value = self.vocab["id2token"][tokenid].split(SPECIAL_COL_SEP)[-1]
                    z3val = map_to_z3var_value(nxt_var_name, value, self.constructor)
                    isvalid = int(z3val.as_long()) in nxt_values
                    #* Domain knowledge: Only check known ports.
                    if 'Pt' in nxt_var_name \
                        and 60_000 in nxt_values \
                        and value not in cidds_ports_str:
                        isvalid = True
                        
                    if isvalid:
                        valid_tokens.append(tokenid)
                        
                if not valid_tokens:
                    # if nxt_var_name == 'Flags':
                    #     #! Let go the last col to see what happens.
                    #     return col_valid_tokenids
                    self.num_invalid_tokens += 1
                    print(f"!!! No valid tokens found for {nxt_var_name}. \nGenerated:")
                    print(f"{self.parent_node[batch_id]=}")
                    print(f"{z3val=} not in {nxt_values=}")
                    pprint(self._generation_cache[batch_id]['generated'])
                    #* Mark this sample as invalid.
                    self._sample_validities[batch_id] = False
                    #! Force an invalid token to be generated.
                    valid_tokens = col_valid_tokenids
                # else:
                #     print(f"{nxt_var_name}: {len(valid_tokens)=}")
                return valid_tokens
            else:
                #& Numeric variable with bounds.  
                assert domain.bounds, f"{nxt_var_name} has no values or bounds in its domain."
                # lb, ub = self._solve_for_minmax(nxt_var_name, tuple(generated))
                
                child_nodes = list(self.trie.successors(self.parent_node[batch_id]))
                assert len(child_nodes) == 1, f"More than one child node: {self.parent_node[batch_id]=}"
                childid = next(self.trie.successors(self.parent_node[batch_id]))
                assert 'bounds' in self.trie.nodes[childid], f"Child node {childid} has no bounds."
                lb, ub = self.trie.nodes[childid]['bounds']

                # print(f"\tLimits for {nxt_var_name}: {lb} ≤ {z3var} ≤ {ub}")
                #TODO: Check if lb==ub -> fast forward generation.
                #* Tokenize lb and ub in the same way as the generated values.
                transform_data = self.col_transform_data[nxt_var_name]
                if transform_data['mx_sig'] < 0:
                    #* Integer
                    total_digits = transform_data['zfill']
                    #* Convert to string and pad with zeros.
                    lb_str = str(int(lb)).zfill(total_digits)
                    ub_str = str(int(ub)).zfill(total_digits)
                    integers = total_digits
                    decimals = 0
                else:
                    #* Floats
                    total_digits = transform_data['ljust'] - 1
                    integers = transform_data['mx_sig']
                    decimals = total_digits - integers
                    lb_str = str(round(lb, decimals))
                    ub_str = str(round(ub, decimals))
                    if '.' not in lb_str: lb_str += '.'
                    if '.' not in ub_str: ub_str += '.'
                    lb_str = lb_str.split('.')[0].zfill(integers) + '.' + lb_str.split('.')[1]
                    ub_str = ub_str.split('.')[0].zfill(integers) + '.' + ub_str.split('.')[1]
                    #* Pad with zeros after the decimal point.
                    lb_str = lb_str.ljust(transform_data['ljust'], '0')
                    ub_str = ub_str.ljust(transform_data['ljust'], '0')
                    assert len(lb_str.split('.')[0]) == integers, f"{lb_str=}, {integers=}"
                    assert len(lb_str.split('.')[1]) == decimals, f"{lb_str=}, {decimals=}"
                # print(f"Limits: {lb_str} ≤ {z3var} ≤ {ub_str}")
                
                self._numeric_gen[batch_id]['started'] = True
                self._numeric_gen[batch_id]['allvalid'] = False
                self._numeric_gen[batch_id]['transform_data'] = transform_data
                self._numeric_gen[batch_id]['var_name'] = nxt_var_name
                # self._numeric_gen[batch_id]['rules'] = substituted_rules
                self._numeric_gen[batch_id]['bounds'] = (lb_str, ub_str)
                self._numeric_gen[batch_id]['digits'] = total_digits
                self._numeric_gen[batch_id]['integer'] = integers
                self._numeric_gen[batch_id]['decimals'] = decimals
                self._numeric_gen[batch_id]['nxt_idx'] = -1
        #* End> if col_finished
                
        if self._numeric_gen[batch_id]['started'] and not self._numeric_gen[batch_id]['allvalid']:
            var_name = self._numeric_gen[batch_id]['var_name']
            # print(f"Generating {var_name}: {self._generation_cache[batch_id]['incomplete']}")
            
            self._numeric_gen[batch_id]['nxt_idx'] += 1
            if (self._numeric_gen[batch_id]['decimals'] > 0 and 
                self._numeric_gen[batch_id]['integer'] == self._numeric_gen[batch_id]['nxt_idx']):
                #* Only decimal point is valid.
                valid_tokens_lb = [token for token in col_valid_tokenids 
                                   if self.vocab["id2token"][token].split(SPECIAL_COL_SEP)[-1] == '.']
                return valid_tokens_lb
            
            nxt_idx = self._numeric_gen[batch_id]['nxt_idx']
            #* Check if the last generated digit.
            if nxt_idx > 0:
                lb_str, ub_str = self._numeric_gen[batch_id]['bounds']
                assert lb_str is not None or ub_str is not None
                lb_digit = lb_str[nxt_idx-1] if lb_str is not None else ' ' #* Space is less than any digit.
                ub_digit = ub_str[nxt_idx-1] if ub_str is not None else 'z' #* z is greater than any digit.
                last_digit = generated_val
                if lb_digit == ub_digit:
                    assert last_digit == lb_digit, f"{last_digit=}, {lb_digit=}, {ub_digit=}"
                else:
                    if lb_digit < last_digit < ub_digit:
                        #* No need to check other digits, e.g., 098 < 4XX < 501
                        self._numeric_gen[batch_id]['allvalid'] = True
                        # print(f"\tSkip checks as of {var_name}[{nxt_idx}].")
                        return col_valid_tokenids
                    else:
                        #* At least one of the bounds is equal to the last digit.
                        #* Need to check the next digit.
                        assert lb_digit == last_digit or last_digit == ub_digit, (
                            f"{last_digit=}, {lb_digit=}, {ub_digit=}")
                        #* Check if one of the bounds is discarded already.
                        if lb_digit != ' ' and ub_digit != 'z':
                            if lb_digit == last_digit:
                                #* Upper bound is guaranteed to be valid, thus ignored.
                                self._numeric_gen[batch_id]['bounds'] = (lb_str, None)
                                # print(f"\tDiscard upper bound for {var_name}[{nxt_idx}].")
                            else:
                                #* Lower bound is guaranteed to be valid, thus ignored.
                                self._numeric_gen[batch_id]['bounds'] = (None, ub_str)
                                # print(f"\tDiscard lower bound for {var_name}[{nxt_idx}].")
            
            lb_str, ub_str = self._numeric_gen[batch_id]['bounds']
            # print(f"Checking: {lb_str} ≤ {var_name}[{nxt_idx}] ≤ {ub_str}")
            invalid_values = []
            valid_tokens_lb = []
            if lb_str is not None:
                lb_digit = lb_str[nxt_idx]
                for tokenid in col_valid_tokenids:
                    value = self.vocab["id2token"][tokenid].split(SPECIAL_COL_SEP)[-1]
                    if value >= lb_digit:
                        valid_tokens_lb.append(tokenid)
                    else:
                        invalid_values.append(value)
            if valid_tokens_lb:
                col_valid_tokenids = valid_tokens_lb
                
            valid_tokens = []
            if ub_str is not None:
                ub_digit = ub_str[nxt_idx]
                for tokenid in col_valid_tokenids:
                    value = self.vocab["id2token"][tokenid].split(SPECIAL_COL_SEP)[-1]
                    if value <= ub_digit:
                        valid_tokens.append(tokenid)
                    else:
                        invalid_values.append(value)
            else:
                valid_tokens = col_valid_tokenids
            
            # print(f"\t{len(valid_tokens)}/{len(col_valid_tokens)} valid tokens for {var_name}[{nxt_idx}].")
            # print(f"\tInvalid tokens: {invalid_values}")
            assert valid_tokens, f"No valid tokens for {var_name} at index {nxt_idx}."
            return valid_tokens

        return col_valid_tokenids

    def _process_seed_input(
        self, seed_input: Union[pd.DataFrame, Dict[str, Any]]
    ) -> torch.Tensor:
        # TODO: The heuristic of choosing the valid columns shouldn't contradict
        # with the `first_col_type` argument of `data_utils.process_data`.`
        if isinstance(seed_input, pd.DataFrame):
            input_cols = seed_input.columns
        elif isinstance(seed_input, dict):
            input_cols = seed_input.keys()
        else:
            raise ValueError(f"Unknown seed_input type: {type(seed_input)}...")

        valid_cols = []
        for col in self.columns:
            if col not in input_cols:
                break
            valid_cols.append(col)

        if isinstance(seed_input, dict):
            seed_input = pd.DataFrame.from_dict({0: seed_input}, orient="index")

        seed_input = seed_input[valid_cols]

        seed_data, _ = process_data(
            df=seed_input, col_transform_data=self.col_transform_data
        )
        seed_data = make_dataset(seed_data, self.vocab, mask_rate=0, affix_eos=False)

        generated = torch.tensor(seed_data["input_ids"])

        if len(generated.shape) == 1:
            generated = generated.unsqueeze(0)

        return generated

    def sample_tabular(
        self,
        n_samples: int,
        check_rules: bool = False,
        gen_batch: Optional[int] = 128,
        device: Optional[str] = "cuda",
        seed_input: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        constrain_tokens_gen: Optional[bool] = True,
        validator: Optional[ObservationValidator] = None,
        continuous_empty_limit: int = 10,
        suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        **generate_kwargs,
    ) -> pd.DataFrame:
        device = torch.device(device)

        if self.model.device != device:
            self.model = self.model.to(device)

        self.model.eval()
        synth_df = []

        if seed_input is None:
            generated = torch.tensor(
                #* Insert beginning of sentence token
                [self.vocab["token2id"][SpecialTokens.BOS] for _ in range(1)]
            ).unsqueeze(0)
        else:
            #* Processing the prompt/instruction/prefix input.
            generated = self._process_seed_input(seed_input=seed_input)

        generated = generated.to(self.model.device)
        
        if check_rules:
            #* Load rules
            rules: List[sp.Expr] = []
            path = "/home/hh1789/Projects/REaLTabFormer/rules/learned_cicids_8192_checked.pl"
            # path = "/home/hh1789/Projects/REaLTabFormer/rules/learned_cidds_8192_checked.pl"
            with open(f"{path}", 'r') as f:
                for i, line in enumerate(f):
                    expr: sp.Expr = sp.sympify(line.strip())
                    rules.append(expr)
                    print(f"Loaded # of rules:\t{i+1}", end='\r')
            print(f"Loaded # of rules:\t{i+1}")
            #TODO: move to device or load the generated outputs to back to cpu for checking and then back to device?
            # rules = torch.tensor(rules).to(device)

        start_time = perf_counter()
        #! `n_samples` is NOT a hard limit. The actual number samples is not enforeced and
        #! can exceed this specified value.
        with tqdm(total=n_samples) as pbar:
            pbar_num_gen = 0
            num_generated = 0
            empty_limit = continuous_empty_limit

            while num_generated < n_samples:
                self._sample_validities = [True for _ in range(gen_batch)]
                self.parent_node = ['root' for _ in range(gen_batch)]
                
                # https://huggingface.co/docs/transformers/internal/generation_utils
                sample_outputs = self._generate(
                    device=device,
                    as_numpy=True,
                    constrain_tokens_gen=constrain_tokens_gen,
                    inputs=generated,
                    do_sample=True,
                    max_length=self.max_length,
                    num_return_sequences=gen_batch,
                    bos_token_id=self.vocab["token2id"][SpecialTokens.BOS],
                    pad_token_id=self.vocab["token2id"][SpecialTokens.PAD],
                    eos_token_id=self.vocab["token2id"][SpecialTokens.EOS],
                    suppress_tokens=suppress_tokens,
                    forced_decoder_ids=forced_decoder_ids,
                    **generate_kwargs,
                )

                self.total_gen_samples += len(sample_outputs)
                # self.invalid_gen_samples += len(sample_outputs)
                invalid_sample_idxes = [i for i, sampleisvalid in enumerate(self._sample_validities) if not sampleisvalid]

                #? Are the following operations happening on the device?
                try:
                    synth_sample: pd.DataFrame = self._processes_sample(
                        sample_outputs=sample_outputs,
                        vocab=self.vocab,
                        validator=validator,
                    )
                    
                    if check_rules:
                        #* Rule-compliance check.
                        violated_rules = set()
                        print(f"Checking rule-compliance for {len(synth_sample)} samples...")
                        for i, sample in tqdm(synth_sample.iterrows(), total=len(synth_sample)):
                            assignment = {}
                            
                            # #* CIDDS
                            # for key in sample.keys():
                            #     if 'Date' in key or 'Flows' in key:
                            #         continue
                            #     if 'Flags' in key:
                            #         value = cidds_flag_map(sample[key])
                            #     elif 'Proto' in key:
                            #         value = cidds_proto_map(sample[key])
                            #     elif 'IP' in key:
                            #         value = cidds_ip_map(sample[key])
                            #     else:
                            #         value = sample[key]
                            #     var = to_big_camelcase(key)
                            #     assignment[var] = value
                            
                            #* CIC
                            for var in sample.keys():
                                if 'ID' in var or 'IP' in var: 
                                    continue
                                value = sample[var]
                                assignment[var] = value
                            
                            for rule in rules:
                                # print(f"{assignment=}")
                                sat = rule.subs(assignment)
                                if not sat:
                                    violated_rules.add(i)
                                    break
                        #* Remove samples that violate the rules
                        print(f"\nRemoved {len(violated_rules)}/{len(synth_sample)} samples that violated the rules.")
                        synth_sample = synth_sample.drop(violated_rules)
                    else:
                        print(f"\n(Batch) Generated {len(invalid_sample_idxes)}/{len(synth_sample)} invalid samples.")
                        synth_sample = synth_sample.drop(invalid_sample_idxes)
                    
                    empty_limit = continuous_empty_limit
                    # self.invalid_gen_samples -= len(synth_sample)
                    self.invalid_gen_samples += len(invalid_sample_idxes)
                    # print(f"Generated {len(synth_sample)} valid samples.")

                except SampleEmptyError as exc:
                    logging.warning("This batch returned an empty valid synth_sample!")
                    empty_limit -= 1
                    if empty_limit <= 0:
                        raise SampleEmptyLimitError(
                            f"The model has generated empty sample batches for {continuous_empty_limit} consecutive rounds!"
                        ) from exc
                    continue

                num_generated += len(synth_sample)
                synth_df.append(synth_sample)
                print(f"Generated {num_generated}/{n_samples} samples.")

                # Update process bar
                pbar.update(num_generated - pbar_num_gen)
                pbar_num_gen = num_generated
        
        end_time = perf_counter()
        
        synth_df = pd.concat(synth_df).sample(
            n=n_samples, replace=False, random_state=self.random_state
        )
        synth_df = synth_df.reset_index(drop="index")

        print(
            f"Invalid samples {self.invalid_gen_samples}/{self.total_gen_samples}. Sampling efficiency is: {100 * (1 -  self.invalid_gen_samples / self.total_gen_samples):.4f}%"
        )
        print(f"Total time taken: {end_time - start_time:.2f}s")
        print(f"Total invalid tokens generated: {self.num_invalid_tokens}")

        return synth_df

    def predict(
        self,
        data: pd.DataFrame,
        target_col: str,
        target_pos_val: Any = None,
        batch: int = 32,
        obs_sample: int = 30,
        fillunk: bool = True,
        device: str = "cuda",
        disable_progress_bar: bool = True,
        **generate_kwargs,
    ) -> pd.Series:
        """
        fillunk: Fill unknown tokens with the mode of the batch.
        target_pos_val: Categorical value for the positive target. This is produces a
         one-to-many prediction relative to `target_pos_val` for targets that are multi-categorical.
        """
        device = torch.device(device)

        if self.model.device != device:
            self.model = self.model.to(device)

        self.model.eval()

        preds = []
        unk_id = self.vocab["token2id"][SpecialTokens.UNK]

        if target_col and target_col in data.columns:
            data = data.drop(target_col, axis=1)

        if disable_progress_bar:
            datasets.utils.disable_progress_bar()

        for i in range(0, len(data), batch):
            seed_data = self._process_seed_input(data.iloc[i : i + batch])
            if fillunk:
                mode = seed_data.mode(dim=0).values
                seed_data[seed_data == unk_id] = torch.tile(mode, (len(seed_data), 1))[
                    seed_data == unk_id
                ]

            sample_outputs = self._generate(
                device=device,
                do_sample=True,
                num_return_sequences=obs_sample,
                input_ids=seed_data.to(device),
                max_length=self.max_length,
                suppress_tokens=[unk_id],
                **generate_kwargs,
            )

            synth_sample = self._processes_sample(
                sample_outputs=sample_outputs,
                vocab=self.vocab,
                validator=None,
            )
            # Reset the index so that we are sure that
            # the index is monotonically increasing.
            # There could be instances where some generation
            # problems arise for some records, we don't
            # handle that here for now.
            synth_sample.reset_index(drop=True, inplace=True)

            preds.extend(
                synth_sample.groupby(synth_sample.index // obs_sample)[
                    target_col
                ].apply(
                    lambda x: (target_pos_val == x).mean()
                    if target_pos_val is not None
                    else x.mean()
                )
            )

        if disable_progress_bar:
            datasets.utils.enable_progress_bar()

        return pd.Series(preds, index=data.index)


class RelationalSampler(REaLSampler):
    """Sampler class for relational data generation."""

    def __init__(
        self,
        model_type: str,
        model: PreTrainedModel,
        vocab: Dict,
        processed_columns: List,
        max_length: int,
        col_size: int,
        col_idx_ids: Dict,
        columns: List,
        datetime_columns: List,
        column_dtypes: Dict,
        column_has_missing: Dict,
        drop_na_cols: List,
        col_transform_data: Dict,
        in_col_transform_data: Dict,
        random_state: Optional[int] = 1029,
        device="cuda",
    ) -> None:
        super().__init__(
            model_type,
            model,
            vocab,
            processed_columns,
            max_length,
            col_size,
            col_idx_ids,
            columns,
            datetime_columns,
            column_dtypes,
            column_has_missing,
            drop_na_cols,
            col_transform_data,
            random_state,
            device,
        )

        self.output_vocab = self.vocab["decoder"]
        self.in_col_transform_data = in_col_transform_data

    @staticmethod
    def sampler_from_model(rtf_model, device: str = "cuda"):
        device = torch.device(device)

        assert rtf_model.relational_max_length is not None
        assert rtf_model.relational_col_size is not None
        assert rtf_model.col_transform_data is not None
        assert rtf_model.in_col_transform_data is not None

        return RelationalSampler(
            model_type=rtf_model.model_type,
            model=rtf_model.model,
            vocab=rtf_model.vocab,
            processed_columns=rtf_model.processed_columns,
            max_length=rtf_model.relational_max_length,
            col_size=rtf_model.relational_col_size,
            col_idx_ids=rtf_model.col_idx_ids,
            columns=rtf_model.columns,
            datetime_columns=rtf_model.datetime_columns,
            column_dtypes=rtf_model.column_dtypes,
            column_has_missing=rtf_model.column_has_missing,
            drop_na_cols=rtf_model.drop_na_cols,
            col_transform_data=rtf_model.col_transform_data,
            in_col_transform_data=rtf_model.in_col_transform_data,
            random_state=rtf_model.random_state,
            device=device,
        )

    def sample_relational(
        self,
        input_unique_ids: Union[pd.Series, List],
        input_df: Optional[pd.DataFrame] = None,
        input_ids: Optional[torch.tensor] = None,
        gen_batch: Optional[int] = 128,
        device: Optional[str] = "cuda",
        constrain_tokens_gen: Optional[bool] = True,
        validator: Optional[ObservationValidator] = None,
        continuous_empty_limit: Optional[int] = 10,
        suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        related_num: Optional[Union[int, List[int]]] = None,
        **generate_kwargs,
    ) -> pd.DataFrame:
        # input_unique_ids: Corresponds to the unique identifier
        # that will be used to link the input
        # data to the generated values.
        device = torch.device(device)

        if self.model.device != device:
            self.model = self.model.to(device)

        self.model.eval()

        input_unique_ids = list(input_unique_ids)

        if input_ids is not None:
            if isinstance(related_num, str):
                warnings.warn(
                    f"The input provided is ids so related_num={related_num} is ignored."
                )
                related_num = None

            generate_kwargs.update(self._get_min_max_length(related_num))

            assert len(input_unique_ids) == len(input_ids)
            input_ids = input_ids.to(device)

            samples = self._generate(
                device=device,
                as_numpy=True,
                constrain_tokens_gen=constrain_tokens_gen,
                inputs=input_ids,
                # num_return_sequences=gen_batch,
                do_sample=True,
                forced_decoder_ids=forced_decoder_ids,
                suppress_tokens=suppress_tokens,
                **generate_kwargs,
            )
        elif input_df is not None:
            assert len(input_unique_ids) == input_df.shape[0]

            # Create a fixed-size matrix to store the data filled with [PAD] token ids.
            samples = np.ones((input_df.shape[0], self.max_length))
            samples *= self.vocab["decoder"]["token2id"][SpecialTokens.PAD]
            start = 0

            if isinstance(related_num, str) and related_num in input_df.columns:
                init_min_max_length = self._get_min_max_length(
                    input_df[related_num].max()
                )

                if init_min_max_length["max_length"] > samples.shape[1]:
                    samples = np.ones(
                        (input_df.shape[0], init_min_max_length["max_length"])
                    )
                    samples *= self.vocab["decoder"]["token2id"][SpecialTokens.PAD]

                _input_unique_ids = []
                input_df = input_df.copy()

                # Make sure that we couple the `input_unique_ids` with
                # its intended data when sorting and grouping.
                input_df[NQ_COL] = input_unique_ids
                input_df.sort_values(related_num, ascending=True, inplace=True)

                for related_num, _input_df in input_df.groupby(related_num):
                    _input_unique_ids.append(_input_df.pop(NQ_COL))
                    generate_kwargs.update(self._get_min_max_length(related_num))

                    for _samples in self._sample_input_batch(
                        input_df=_input_df,
                        gen_batch=gen_batch,
                        device=device,
                        constrain_tokens_gen=constrain_tokens_gen,
                        suppress_tokens=suppress_tokens,
                        forced_decoder_ids=forced_decoder_ids,
                        **generate_kwargs,
                    ):
                        end = start + len(_samples)

                        samples[start:end, : _samples.shape[1]] = _samples
                        start = end

                input_unique_ids = pd.concat(_input_unique_ids)
            else:
                generate_kwargs.update(self._get_min_max_length(related_num))

                if generate_kwargs["max_length"] > samples.shape[1]:
                    # Create a fixed-size matrix to store the data filled with [PAD] token ids.
                    samples = np.ones(
                        (input_df.shape[0], generate_kwargs["max_length"])
                    )
                    samples *= self.vocab["decoder"]["token2id"][SpecialTokens.PAD]

                for _samples in self._sample_input_batch(
                    input_df=input_df,
                    gen_batch=gen_batch,
                    device=device,
                    constrain_tokens_gen=constrain_tokens_gen,
                    suppress_tokens=suppress_tokens,
                    forced_decoder_ids=forced_decoder_ids,
                    **generate_kwargs,
                ):
                    end = start + len(_samples)

                    samples[start:end, : _samples.shape[1]] = _samples
                    start = end
        else:
            raise ValueError("Either `input_ids` or `input_df` must not be None.")

        synth_df = self._processes_sample(
            sample_outputs=samples,
            vocab=self.vocab["decoder"],
            relate_ids=input_unique_ids,
            validator=validator,
        )

        return synth_df

    def _get_min_max_length(self, related_num):
        # The `min_length = 2` corresponds to the ([EOS], [BOS])
        # sequence that is used in the encoder-decoder
        # model. This is why a related num of zero will have a
        # max_length of (2 + 1) because we expect the next token
        # in this sequence should be [EOS]. In the case where
        # the `related_num > 0` we add `2` to col_size to account
        # for the [BMEM] and the [EMEM] tokens.
        min_length = 2
        max_length = self.max_length

        if related_num is not None:
            if related_num >= 0:
                min_length = min_length + ((self.col_size + 2) * related_num)
                max_length = min_length + 1
            else:
                raise ValueError(
                    "The `related_num` must be greater than or equal to zero."
                )

        return dict(min_length=min_length, max_length=max_length)

    def _sample_input_batch(
        self,
        input_df: Optional[pd.DataFrame] = None,
        gen_batch: Optional[int] = 128,
        device: Optional[str] = "cuda",
        constrain_tokens_gen: Optional[bool] = True,
        suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        **generate_kwargs,
    ):
        # Let apply processing if `input_df` is given.
        input_df, _ = process_data(
            input_df, col_transform_data=self.in_col_transform_data
        )

        # Load the dataframe into a HuggingFace Dataset
        dataset = make_dataset(input_df, self.vocab["encoder"])

        input_loader = DataLoader(
            dataset, batch_size=gen_batch, collate_fn=DefaultDataCollator()
        )
        loader_iters = iter(input_loader)

        for batch in tqdm(loader_iters):
            input_ids = batch["input_ids"].to(device)

            _samples = self._generate(
                device=device,
                as_numpy=True,
                constrain_tokens_gen=constrain_tokens_gen,
                inputs=input_ids,
                # num_return_sequences=gen_batch,
                do_sample=True,
                forced_decoder_ids=forced_decoder_ids,
                suppress_tokens=suppress_tokens,
                **generate_kwargs,
            )

            yield _samples

    def _get_relational_col_idx_ids(self, len_ids: int) -> List:
        """This method returns the true index given the generation step `i`.

        col_size: The expected number of variables for a single observation.
            This is equal to the number of columns.

        ### Generating constrained tokens per step
        ```
            1 -> BOS
            2 -> BMEM or EOS
            3 -> col 0
            ...
            3 + col_size -> col col_size - 1
            3 + col_size + 1 -> EMEM
            3 + col_size + 2 -> BMEM or EOS
            3 + col_size + 3 -> col 0
        ```
        """
        if len_ids == 0:
            return_ids = [self.vocab["decoder"]["token2id"][SpecialTokens.EOS]]
        elif len_ids == 1:
            # This is the decoder_start_token_id and we should generate the [BOS] token next.
            return_ids = [self.vocab["decoder"]["token2id"][SpecialTokens.BOS]]
        else:
            # Adjust such that idx = 0 produces either the [BMEM] or the [EOS] tokens.
            idx = len_ids - 2

            # Number of columns in the data for each observation
            # self.col_size = self.relational_col_size

            # Number of pads between observations [EMEM], ([BMEM] | [EOS])
            num_pads = 2

            # Derive `col_idx` such that at idx == 0, it will return -1 which
            # maps to [[BMEM] or [EOS]]. Then idx == 1 will generate the first
            # valid token, ..., then at idx == col_size, will generate the
            # last valid token. At idx == col_size + 1, it will generate col_size
            # which should generate the [[EMEM]] token.
            col_idx = (idx % (self.col_size + num_pads)) - 1
            assert -1 <= col_idx <= self.col_size

            if col_idx < 0:
                col_idx = -1
            elif col_idx >= self.col_size:
                col_idx = -2

            return_ids = self.col_idx_ids[col_idx]

        return return_ids

    def _prefix_allowed_tokens_fn(self, batch_id, input_ids) -> List:
        # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.prefix_allowed_tokens_fn
        # For the relational data, len(input_ids) == 2 -> [EOS, BOS]
        return self._get_relational_col_idx_ids(len(input_ids))
