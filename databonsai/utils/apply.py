from tqdm import tqdm
from databonsai.categorize import BaseCategorizer
from databonsai.transform import BaseTransformer, DecomposeTransformer
from typing import List, Callable, get_origin
import inspect


def apply_to_column(
    input_column: List,
    output_column: List,
    func: Callable,
    start_idx: int = 0,
) -> int:
    """
    Apply a function to each value in a column of a DataFrame or a normal Python list, starting from a specified index.

    Parameters:
        input_column (List): The column of the DataFrame or a normal Python list to apply the function to.
        output_column (List): A list to store the processed values. The function will mutate this list in-place.
        func (callable): The function to apply to each value in the column.
                         The function should take a single value as input and return a single value.
        start_idx (int, optional): The index from which to start applying the function. Default is 0.

    Returns:
        int: The index of the last successfully processed value.

    """
    if len(input_column) == 0:
        raise ValueError("Input input_column is empty.")

    if start_idx >= len(input_column):
        raise ValueError(
            f"start_idx ({start_idx}) is greater than or equal to the length of the input_column ({len(input_column)})."
        )

    if len(output_column) > len(input_column):
        raise ValueError(
            f"The length of the output_column ({len(output_column)}) is greater than the length of the input_column ({len(input_column)})."
        )

    success_idx = start_idx

    if isinstance(func.__self__, BaseCategorizer):
        desc = "Categorizing"
    elif isinstance(func.__self__, BaseTransformer):
        desc = "Transforming"
    else:
        desc = "Processing"

    try:
        for idx, value in enumerate(
            tqdm(input_column[start_idx:], desc=desc, unit="row"), start=start_idx
        ):
            result = func(value)

            if idx >= len(output_column):
                output_column.append(result)
            else:
                output_column[idx] = result

            success_idx = idx + 1
    except Exception as e:
        print(f"Error occurred at index {success_idx}: {str(e)}")
        print(f"Processing stopped at index {success_idx - 1}")
        return success_idx

    return success_idx


def apply_to_column_batch(
    input_column: List,
    output_column: List,
    func: Callable,
    batch_size: int = 5,
    start_idx: int = 0,
) -> int:
    """
    Apply a function to each batch of values in a column of a DataFrame or a normal Python list, starting from a specified index.

    Parameters:
        input_column (List): The column of the DataFrame or a normal Python list to apply the function to.
        output_column (List): A list to store the processed values. The function will mutate this list in-place.
        func (callable): The function to apply to each batch of values in the column.
                         The function should take a list of values as input and return a list of processed values.
        batch_size (int, optional): The size of each batch. Default is 5.
        start_idx (int, optional): The index from which to start applying the function. Default is 0.

    Returns:
        tuple: A tuple containing two elements:
               - success_idx (int): The index of the last successfully processed batch.

    """
    if len(input_column) == 0:
        raise ValueError("Input input_column is empty.")

    if start_idx >= len(input_column):
        raise ValueError(
            f"start_idx ({start_idx}) is greater than or equal to the length of the input_column ({len(input_column)})."
        )

    if len(output_column) > len(input_column):
        raise ValueError(
            f"The length of the output_column list ({len(output_column)}) is greater than the length of th input_column ({len(column)})."
        )

    if not inspect.signature(func).parameters:
        raise TypeError("The provided function does not take any arguments.")

    # Ensure func is a batch function that takes a list
    first_param = list(inspect.signature(func).parameters.values())[0]
    param_annotation = first_param.annotation
    origin = get_origin(param_annotation)

    if origin is not get_origin(List):
        raise TypeError(
            "The provided function does not take a list or pandas.Series as input."
        )

    success_idx = start_idx

    if isinstance(func.__self__, BaseCategorizer):
        desc = "Categorizing"
    elif isinstance(func.__self__, BaseTransformer):
        desc = "Transforming"
    else:
        desc = "Processing"

    try:
        for i in tqdm(
            range(start_idx, len(input_column), batch_size), desc=desc, unit="batch"
        ):
            batch_end = min(i + batch_size, len(input_column))
            batch = input_column[i:batch_end]
            batch_result = func(batch)
            if i >= len(output_column):
                output_column.extend(batch_result)
            else:
                output_column[i : i + len(batch_result)] = batch_result
            success_idx = batch_end
    except Exception as e:
        print(f"Error occurred at batch starting at index {success_idx}: {str(e)}")
        print(f"Processing stopped at batch ending at index {success_idx - 1}")
        return success_idx

    return min(success_idx, len(input_column))
