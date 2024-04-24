from tqdm import tqdm
from typing import List, Callable, Union, get_origin
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

    try:
        for idx, value in enumerate(
            tqdm(input_column[start_idx:], desc="Processing data..", unit="row"),
            start=start_idx,
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
        func (callable): The batch function to apply to each batch of values in the column.
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
            f"The length of the output_column list ({len(output_column)}) is greater than the length of th input_column ({len(input_column)})."
        )

    check_func(func)
    success_idx = start_idx
    try:
        for i in tqdm(
            range(start_idx, len(input_column), batch_size),
            desc="Processing data..",
            unit="batch",
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


def apply_to_column_autobatch(
    input_column: List,
    output_column: List,
    func: Callable,
    max_retries: int = 3,
    max_batch_size: int = 5,
    batch_size: int = 2,
    ramp_factor: float = 1.5,
    ramp_factor_decay: float = 0.8,
    reduce_factor: float = 0.5,
    reduce_factor_decay: float = 0.8,
    start_idx: int = 0,
) -> int:
    """
    Apply a function to the input column using adaptive batch processing.

    This function applies a batch processing function to the input column, starting from the
    specified index. It adaptively adjusts the batch size based on the success or failure of
    each batch processing attempt. The function retries failed batches with a reduced batch
    size and gradually decreases the rate of batch size adjustment over time.

    Parameters:
        input_column (List): The input column to be processed.
        output_column (List): The list to store the processed results.
        func (callable): The batch function to apply to each batch of values in the column.
                         The function should take a list of values as input and return a list of processed values.
        max_retries (int): The maximum number of retries for failed batches.
        max_batch_size (int): The maximum allowed batch size.
        batch_size (int): The initial batch size.
        ramp_factor (float): The factor by which the batch size is increased after a successful batch.
        ramp_factor_decay (float): The decay rate for the ramp factor after each successful batch.
        reduce_factor (float): The factor by which the batch size is reduced after a failed batch.
        reduce_factor_decay (float): The decay rate for the reduce factor after each failed batch.
        start_idx (int): The index from which to start processing the input column.

    Returns:
        int: The index of the last successfully processed item in the input column.

    """
    if len(input_column) == 0:
        raise ValueError("Input input_column is empty.")

    if start_idx >= len(input_column):
        raise ValueError(
            f"start_idx ({start_idx}) is greater than or equal to the length of the input_column ({len(input_column)})."
        )

    if len(output_column) > len(input_column):
        raise ValueError(
            f"The length of the output_column list ({len(output_column)}) is greater than the length of the input_column ({len(input_column)})."
        )

    check_func(func)
    success_idx = start_idx
    ramp_factor = ramp_factor
    reduce_factor = reduce_factor

    try:
        remaining_data = input_column[start_idx:]
        processed_results = []
        batch_size = batch_size
        retry_count = 0

        with tqdm(
            total=len(remaining_data), desc="Processing data..", unit="row"
        ) as pbar:
            while len(remaining_data) > 0:
                try:
                    batch_size = min(batch_size, len(remaining_data))
                    batch = remaining_data[:batch_size]
                    batch_results = func(batch)
                    processed_results.extend(batch_results)
                    remaining_data = remaining_data[batch_size:]
                    retry_count = 0

                    # Update progress bar
                    pbar.update(batch_size)

                    # Increase the batch size using the decayed ramp factor
                    batch_size = min(round(batch_size * ramp_factor), max_batch_size)
                    ramp_factor = max(ramp_factor * ramp_factor_decay, 1.0)

                except Exception as e:

                    if retry_count >= max_retries:
                        raise ValueError(
                            f"Processing failed after {max_retries} retries. Error: {str(e)}"
                        )
                    retry_count += 1

                    # Decrease the batch size using the decayed reduce factor
                    batch_size = max(round(batch_size * reduce_factor), 1)
                    print(f"Retrying with smaller batch size: {batch_size}")
                    reduce_factor *= reduce_factor_decay

        output_column[start_idx:] = processed_results
        success_idx = start_idx + len(processed_results)
    except Exception as e:
        print(f"Error occurred at batch starting at index {success_idx}: {str(e)}")
        print(f"Processing stopped at batch ending at index {success_idx - 1}")
        return success_idx

    return min(success_idx, len(input_column))


def check_func(func):
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
